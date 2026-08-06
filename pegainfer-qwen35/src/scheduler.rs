//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - controller-owned `RequestKv` plus recurrent state (hybrid attention)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)

mod plan;

use std::sync::OnceLock;
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use log::debug;
use log::info;
use log::warn;
use pegainfer_core::engine::EngineHandle as SchedulerHandle;
use pegainfer_core::engine::FinishReason;
use pegainfer_core::engine::GenerateRequest as SchedulerRequest;
use pegainfer_core::engine::KvCapacity;
use pegainfer_core::engine::LoadSnapshot;
use pegainfer_core::engine::SubmittedRequest;
use pegainfer_core::engine::TokenEvent;
use pegainfer_core::engine::TokenLogprob;
use pegainfer_core::engine::TokenSink;
use pegainfer_core::engine::panic_message;
use pegainfer_core::sampler::SamplingParams;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kv_cache::KvCacheManager;
use pegainfer_kv_cache::RequestKv;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;
use tokio::sync::watch;

use self::plan::ActiveDecodeState;
use self::plan::ActiveKvBudget;
use self::plan::ExecutionPlan;
use self::plan::PrefillKvBudget;
use self::plan::PrefillQueueState;
use self::plan::RejectReason;
use self::plan::admit_pending_requests;
use self::plan::choose_prefill_budget;
use self::plan::compaction_after_retire;
use self::plan::max_kv_tokens;
use self::plan::plan_prefill_chunks;
use self::plan::prefilling_future_pages;
use self::plan::slot_for_new_request;
use crate::Qwen35SchedulerPolicy;
use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::executor::DecodeRequestResult;
use crate::executor::DecodeResult;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::RequestId;
use crate::logprobs::snapshot_requested_logprobs;
use crate::prefix_cache::Qwen35PrefixCache;
use crate::prefix_cache::RecurrentStateStore;
use crate::recurrent_state::RecurrentState;
use crate::tp_executor::Qwen35TpExecutor;
use crate::tp_executor::TpDecodeStepItem;
use crate::tp_executor::TpPrefillChunkItem;
use crate::weights::Qwen35Model;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
struct ActiveRequest35 {
    request_id: Option<String>,
    token_tx: TokenSink,
    backend_state: ActiveBackendState,
    last_token: u32,
    generated_count: usize,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    logprobs: usize,
}

/// A request whose prompt is being prefilled across multiple scheduler steps.
/// It owns its growing KV and recurrent state until the prompt is exhausted,
/// at which point it is promoted into the decode batch.
struct PrefillingRequest35 {
    req: SchedulerRequest,
    backend_state: PrefillBackendState,
    /// Prompt tokens prefilled so far.
    cursor: usize,
    /// Tokens to prefill in the step currently scheduled (set by `take_prefill_chunks`).
    step_chunk: usize,
}

enum ActiveBackendState {
    Single {
        kv: Box<RequestKv>,
        /// Index into `BatchDecodeGraphState.slot_states`.
        graph_slot_idx: usize,
    },
    Tp {
        request_id: RequestId,
    },
}

enum PrefillBackendState {
    Single {
        kv: Box<RequestKv>,
        rec: RecurrentState,
    },
    Tp {
        request_id: RequestId,
    },
}

fn active_request_kv(request: &mut ActiveRequest35) -> Option<&mut RequestKv> {
    match &mut request.backend_state {
        ActiveBackendState::Single { kv, .. } => Some(kv),
        ActiveBackendState::Tp { .. } => None,
    }
}

/// Roll back a set of requests scheduled by the current scheduler step.
fn revert_scheduled_requests<'a>(
    kv_cache: &Qwen35PrefixCache,
    requests: impl IntoIterator<Item = &'a mut RequestKv>,
) {
    for request in requests {
        if let Err(error) = kv_cache.revert_schedule(request) {
            warn!("failed to revert Qwen3.5 scheduler KV schedule: {error}");
        }
    }
}

pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

/// Env-gated per-step ITL diagnostics (issue #470). When `PEGAINFER_ITL_DEBUG`
/// is set, the scheduler emits one `ITL_STEP` line per executed step, tagging
/// the plan kind, the *actual* prefill-chunk token count run this step, the
/// number of active decode rows frozen behind it, and the CPU wall-time the
/// step took. This lets the mixed-load bench attribute a background decode
/// stall to the specific steps that truly ran a prefill chunk, instead of the
/// coarse `[submit, last-token]` injection window that spans every step of a
/// long chunked prefill. Off by default: no cost on the normal bench path.
fn itl_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PEGAINFER_ITL_DEBUG").is_some())
}

/// Monotonic microseconds since the first ITL step, so `ITL_STEP` timestamps
/// are correlatable within one process run (paired with wall-clock epoch us).
fn itl_debug_mono_us() -> u128 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_micros()
}

fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64())
}

// ── Entry point ─────────────────────────────────────────────────────────

pub fn start_with_capacity(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
) -> Result<SchedulerHandle> {
    start_with_capacity_and_policy(
        model,
        seed,
        max_batch,
        max_prefill_tokens,
        Qwen35SchedulerPolicy::Off,
    )
}

pub(crate) fn start_with_capacity_and_policy(
    model: Qwen35Model,
    seed: u64,
    max_batch: usize,
    max_prefill_tokens: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let backend = SingleGpuBackend::new(model, max_batch)?;
    // Static instance cap for the vLLM bridge's max_model_len. Live admission
    // still uses the current page budget inside the scheduler loop.
    let total_blocks = backend.kv_cache.pool().max_request_blocks();
    let kv_total_blocks = total_blocks as u64;
    let block_size = backend.kv_cache.pool().block_size();
    let servable = servable_len(
        backend.model.config().max_position_embeddings,
        total_blocks,
        block_size,
    );
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (startup_tx, startup_rx) = std_mpsc::channel();
    let (load_tx, load_rx) = watch::channel(LoadSnapshot {
        kv_total_blocks,
        ..LoadSnapshot::default()
    });

    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || match bind_model_thread(backend.model()) {
            Ok(_guard) => {
                let _ = startup_tx.send(Ok(()));
                scheduler_loop(
                    SchedulerBackend::Single(backend),
                    submit_rx,
                    seed,
                    max_prefill_tokens,
                    scheduler_policy,
                    load_tx,
                );
            }
            Err(err) => {
                let _ = startup_tx.send(Err(err));
            }
        })
        .expect("failed to spawn Qwen3.5 scheduler thread");

    let Ok(startup) = startup_rx.recv() else {
        let panic_note = match join_handle.join() {
            Err(panic) => format!(" (thread panicked: {})", panic_message(panic.as_ref())),
            Ok(()) => String::new(),
        };
        anyhow::bail!("Qwen3.5 scheduler exited during startup{panic_note}");
    };
    if let Err(err) = startup {
        let _ = join_handle.join();
        return Err(err);
    }
    Ok(
        SchedulerHandle::new_with_join_handle(submit_tx, join_handle)
            .with_servable_len(servable)
            .with_kv_capacity(KvCapacity {
                total_blocks,
                block_size,
            })
            .with_load_watch(load_rx),
    )
}

pub(crate) fn start_tp_with_capacity(
    model_path: &str,
    seed: u64,
    device_ordinals: &[usize],
    max_batch: usize,
    max_prefill_tokens: usize,
    prefix_snapshot_bytes: usize,
) -> Result<SchedulerHandle> {
    assert!(
        max_prefill_tokens > 0,
        "max_prefill_tokens must be positive: a zero budget can never schedule a prefill chunk"
    );
    let backend = TpSchedulerBackend::new(
        model_path,
        device_ordinals,
        max_batch,
        max_prefill_tokens,
        prefix_snapshot_bytes,
    )?;
    let servable = servable_len(
        backend.max_position_embeddings(),
        backend.capacity_pages_for_requests(),
        backend.page_size(),
    );
    let kv_capacity = KvCapacity {
        total_blocks: backend.capacity_pages_for_requests(),
        block_size: backend.page_size(),
    };

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (load_tx, load_rx) = watch::channel(LoadSnapshot {
        kv_total_blocks: kv_capacity.total_blocks as u64,
        ..LoadSnapshot::default()
    });
    let join_handle = thread::Builder::new()
        .name("scheduler-qwen35-tp".into())
        .spawn(move || {
            scheduler_loop(
                SchedulerBackend::Tp(backend),
                submit_rx,
                seed,
                max_prefill_tokens,
                Qwen35SchedulerPolicy::Off,
                load_tx,
            );
        })
        .expect("failed to spawn Qwen3.5 TP scheduler thread");

    Ok(
        SchedulerHandle::new_with_join_handle(submit_tx, join_handle)
            .with_servable_len(servable)
            .with_kv_capacity(kv_capacity)
            .with_load_watch(load_rx),
    )
}

struct SingleGpuBackend {
    model: Qwen35Model,
    kv_cache: Qwen35PrefixCache,
    recurrent_store: RecurrentStateStore,
    graph_state: BatchDecodeGraphState,
}

// One instance per scheduler; the size asymmetry costs nothing here.
#[allow(clippy::large_enum_variant)]
enum SchedulerBackend {
    Single(SingleGpuBackend),
    Tp(TpSchedulerBackend),
}

struct TpSchedulerBackend {
    executor: Qwen35TpExecutor,
    next_request_id: u64,
}

impl SingleGpuBackend {
    fn new(model: Qwen35Model, max_batch: usize) -> Result<Self> {
        anyhow::ensure!(max_batch > 0, "Qwen3.5 max_batch must be > 0");
        let manager =
            KvCacheManager::from_buffer(model.kv_buffer().clone(), model.kv_buffer().num_blocks())?;
        let kv_cache = Qwen35PrefixCache::new(manager, model.prefix_snapshot_slots())?;
        let recurrent_store = RecurrentStateStore::new(
            model.device_ctx(),
            model.config(),
            model.prefix_snapshot_slots(),
        )?;
        debug_assert_eq!(recurrent_store.len(), kv_cache.snapshot_slots());
        let graph_capacity = crate::batch_decode_graph::bucket_for(max_batch);
        let graph_state = model.create_batch_decode_graph_state_with_capacity(
            graph_capacity,
            kv_cache.pool().total_blocks(),
            kv_cache.pool().padding_block_id(),
        )?;
        info!(
            "Qwen3.5 prefix cache: enabled={}, snapshot_slots={}",
            kv_cache.enabled(),
            kv_cache.snapshot_slots()
        );
        Ok(Self {
            model,
            kv_cache,
            recurrent_store,
            graph_state,
        })
    }

    fn model(&self) -> &Qwen35Model {
        &self.model
    }

    fn max_batch(&self) -> usize {
        // #470: admit the requested `--max-batch`, which may sit below the loaded
        // graph bucket (e.g. 5 on bucket 8); never exceed the physical slots.
        self.model
            .decode_admission_batch
            .min(self.graph_state.slot_states.len())
            .max(1)
    }

    fn page_size(&self) -> usize {
        self.kv_cache.pool().block_size()
    }

    fn available_pages(&self) -> usize {
        self.kv_cache.pool().available_blocks()
    }

    fn capacity_pages_for_requests(&self) -> usize {
        self.kv_cache.pool().max_request_blocks()
    }

    fn max_position_embeddings(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    fn alloc_recurrent(&self) -> Result<RecurrentState> {
        RecurrentState::new(self.model.device_ctx(), self.model.config())
    }

    fn alloc_prefill_state(
        &mut self,
        req: &SchedulerRequest,
    ) -> Result<(PrefillBackendState, usize)> {
        let mut rec = self.alloc_recurrent()?;
        let (mut kv, restore) = self.kv_cache.begin_request(
            &req.prompt_tokens,
            req.max_tokens,
            req.lora_adapter.as_deref(),
            !req.echo,
        )?;
        let cached_tokens = if let Some(restore) = restore {
            if let Err(error) = self.recurrent_store.restore(
                self.model.device_ctx(),
                restore.recurrent_slot(),
                &mut rec,
            ) {
                let _ = self.kv_cache.release_request(&mut kv);
                return Err(error);
            }
            match self.kv_cache.finish_restore(&kv, restore, &[rec.seq_len]) {
                Ok(tokens) => tokens,
                Err(error) => {
                    let _ = self.kv_cache.release_request(&mut kv);
                    return Err(error);
                }
            }
        } else {
            0
        };
        Ok((
            PrefillBackendState::Single {
                kv: Box::new(kv),
                rec,
            },
            cached_tokens,
        ))
    }

    fn batch_prefill_logits(&self, chunk: &mut ScheduledChunk) -> Result<HiddenStates> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU prefill received TP chunk state");
        };
        for (scheduled, (kv, window)) in kvs.iter_mut().zip(&chunk.windows).enumerate() {
            if let Err(error) = self.kv_cache.schedule_prefill(kv, window.len()) {
                revert_scheduled_requests(&self.kv_cache, kvs.iter_mut().take(scheduled));
                return Err(error);
            }
        }
        let views = kvs
            .iter()
            .zip(&chunk.windows)
            .map(|(kv, window)| self.kv_cache.prefill_view(kv, window.len()))
            .collect::<Vec<_>>();
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let result = self.model.batch_prefill_logits(
            &window_refs,
            &views,
            &mut rec_refs,
            self.kv_cache.buffer(),
        );
        if result.is_err() {
            revert_scheduled_requests(&self.kv_cache, kvs);
        }
        result
    }

    fn unified_step(
        &mut self,
        chunk: &mut ScheduledChunk,
        active: &mut [ActiveRequest35],
    ) -> Result<crate::unified_forward::UnifiedStepOutput> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU unified step received TP chunk state");
        };
        for (scheduled_prefills, (kv, window)) in kvs.iter_mut().zip(&chunk.windows).enumerate() {
            if let Err(error) = self.kv_cache.schedule_prefill(kv, window.len()) {
                revert_scheduled_requests(&self.kv_cache, kvs.iter_mut().take(scheduled_prefills));
                return Err(error);
            }
        }
        let prefill_views = kvs
            .iter()
            .zip(&chunk.windows)
            .map(|(kv, window)| self.kv_cache.prefill_view(kv, window.len()))
            .collect::<Vec<_>>();
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        for (scheduled_decodes, req) in active.iter_mut().enumerate() {
            let ActiveBackendState::Single { kv, .. } = &mut req.backend_state else {
                panic!("single-GPU unified step received TP active state")
            };
            if let Err(error) = self.kv_cache.schedule_decode(kv) {
                revert_scheduled_requests(&self.kv_cache, kvs);
                revert_scheduled_requests(
                    &self.kv_cache,
                    active
                        .iter_mut()
                        .take(scheduled_decodes)
                        .filter_map(active_request_kv),
                );
                return Err(error);
            }
        }
        let decode_views = active
            .iter()
            .map(|r| match &r.backend_state {
                ActiveBackendState::Single { kv, .. } => self.kv_cache.decode_view(kv),
                ActiveBackendState::Tp { .. } => {
                    panic!("single-GPU unified step received TP active state")
                }
            })
            .collect::<Vec<_>>();
        let result = self.model.unified_step(
            &window_refs,
            &prefill_views,
            &mut rec_refs,
            &decode_tokens,
            &decode_views,
            self.kv_cache.buffer(),
            &mut self.graph_state,
        );
        if result.is_err() {
            revert_scheduled_requests(&self.kv_cache, kvs);
            revert_scheduled_requests(
                &self.kv_cache,
                active.iter_mut().filter_map(active_request_kv),
            );
        }
        result
    }

    fn decode_graph(&mut self, active: &mut [ActiveRequest35]) -> Result<()> {
        let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        for (scheduled, req) in active.iter_mut().enumerate() {
            let ActiveBackendState::Single { kv, .. } = &mut req.backend_state else {
                panic!("single-GPU decode received TP active state")
            };
            if let Err(error) = self.kv_cache.schedule_decode(kv) {
                revert_scheduled_requests(
                    &self.kv_cache,
                    active
                        .iter_mut()
                        .take(scheduled)
                        .filter_map(active_request_kv),
                );
                return Err(error);
            }
        }
        let views = active
            .iter()
            .map(|r| match &r.backend_state {
                ActiveBackendState::Single { kv, .. } => self.kv_cache.decode_view(kv),
                ActiveBackendState::Tp { .. } => panic!("single-GPU decode received TP state"),
            })
            .collect::<Vec<_>>();
        let result = self.model.batch_decode_graph(
            &token_ids,
            &views,
            self.kv_cache.buffer(),
            &mut self.graph_state,
        );
        if result.is_err() {
            revert_scheduled_requests(
                &self.kv_cache,
                active.iter_mut().filter_map(active_request_kv),
            );
        }
        result
    }

    fn apply_prefill(&mut self, chunk: &mut ScheduledChunk, tokens: &[u32]) -> Result<()> {
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU commit received TP chunk state")
        };
        for (i, (kv, rec)) in kvs.iter_mut().zip(recs.iter()).enumerate() {
            let is_final = chunk.ends[i] == chunk.reqs[i].prompt_tokens.len();
            let boundary = self
                .kv_cache
                .apply_prefill(kv, is_final.then_some(tokens[i]))?;
            anyhow::ensure!(
                rec.seq_len == boundary,
                "Qwen3.5 prefill apply position mismatch: kv={boundary}, recurrent={}",
                rec.seq_len
            );
            if let Some(reservation) = self.kv_cache.reserve_prefix(kv, boundary)? {
                if let Err(error) = self.recurrent_store.save(
                    self.model.device_ctx(),
                    reservation.recurrent_slot(),
                    rec,
                ) {
                    self.kv_cache.abort_prefix(reservation);
                    return Err(error);
                }
                self.kv_cache.publish_prefix(kv, reservation);
            }
        }
        Ok(())
    }

    fn apply_decode(&self, active: &mut [ActiveRequest35], tokens: &[u32]) -> Result<()> {
        anyhow::ensure!(active.len() == tokens.len(), "decode apply row mismatch");
        for (req, &token) in active.iter_mut().zip(tokens) {
            let ActiveBackendState::Single { kv, .. } = &mut req.backend_state else {
                anyhow::bail!("single-GPU decode apply received TP state")
            };
            self.kv_cache.apply_decode(kv, token)?;
        }
        Ok(())
    }

    fn fail_active(&self, active: &mut Vec<ActiveRequest35>, message: &str) {
        for mut req in active.drain(..) {
            if let ActiveBackendState::Single { kv, .. } = &mut req.backend_state {
                if let Err(error) = self.kv_cache.release_request(kv) {
                    warn!("failed to release Qwen3.5 request KV: {error}");
                }
            }
            let _ = req.token_tx.send(TokenEvent::Error {
                message: message.to_string(),
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
        }
    }

    fn log_prefix_cache_stats(&self) {
        let cache = &self.kv_cache;
        let stats = cache.stats();
        info!(
            "Qwen3.5 prefix cache summary: joint_hits={}, hit_tokens={}, kv_only_fallbacks={}, snapshot_misses={}, inserts={}, evictions={}, restore_ms={:.3}, occupancy={}/{}",
            stats.joint_hits,
            stats.joint_hit_tokens,
            stats.kv_only_fallbacks,
            stats.snapshot_misses,
            stats.inserts,
            stats.evictions,
            stats.restore_ns as f64 / 1_000_000.0,
            cache.snapshot_occupancy(),
            cache.snapshot_slots(),
        );
    }

    fn sample_prefill_logits(
        &mut self,
        pending: &[SchedulerRequest],
        logits: &HiddenStates,
        rng: &mut StdRng,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        debug_assert_eq!(
            logits.seq_len,
            pending.len(),
            "Qwen3.5 prefill logits rows must preserve pending request order"
        );
        let requested_logprobs: Vec<usize> = pending.iter().map(|r| r.logprobs).collect();
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &requested_logprobs)?;
        let params_refs: Vec<&SamplingParams> = pending.iter().map(|r| &r.params).collect();
        let sample_seed = rand::RngExt::random(rng);
        let tokens = self.model.select_tokens_from_logits_varied(
            logits,
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = cpu_logits
            .into_iter()
            .enumerate()
            .map(|(i, logits_opt)| {
                logits_opt.and_then(|logits_f32| {
                    pegainfer_sample::token_logprob_from_row(
                        &logits_f32,
                        tokens[i],
                        pending[i].logprobs,
                    )
                })
            })
            .collect();
        Ok((tokens, logprobs))
    }

    fn sample_decode_logits(
        &mut self,
        active: &[ActiveRequest35],
        rng: &mut StdRng,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        let requested_logprobs: Vec<usize> = active.iter().map(|r| r.logprobs).collect();
        let cpu_logits = snapshot_requested_logprobs(
            self.model.device_ctx(),
            &self.graph_state.buffers.logits,
            &requested_logprobs,
        )?;
        let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
        let sample_seed = rand::RngExt::random(rng);
        let tokens = self.model.select_tokens_batch_varied(
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = cpu_logits
            .into_iter()
            .enumerate()
            .map(|(i, logits_opt)| {
                logits_opt.and_then(|logits_f32| {
                    pegainfer_sample::token_logprob_from_row(
                        &logits_f32,
                        tokens[i],
                        active[i].logprobs,
                    )
                })
            })
            .collect();
        Ok((tokens, logprobs))
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.model.is_stop_token(token)
    }

    fn copy_recurrent_to_slot(
        &mut self,
        recurrent: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        self.graph_state
            .copy_state_to_slot(self.model.device_ctx(), recurrent, slot_idx)
    }

    fn compact_slot(&mut self, active: &mut [ActiveRequest35], compaction: plan::SlotCompaction) {
        let src_slot = match active[compaction.moved_to].backend_state {
            ActiveBackendState::Single { graph_slot_idx, .. } => graph_slot_idx,
            ActiveBackendState::Tp { .. } => {
                panic!("single-GPU slot compaction received TP active state")
            }
        };
        debug_assert_eq!(src_slot, compaction.moved_from);

        let ctx = self.model.device_ctx();
        let src = &self.graph_state.slot_states[compaction.moved_from];
        for layer_idx in 0..src.layers.len() {
            let (src_part, dst_part) = if compaction.moved_to < compaction.moved_from {
                let (left, right) = self
                    .graph_state
                    .slot_states
                    .split_at_mut(compaction.moved_from);
                (
                    &right[0].layers[layer_idx],
                    &mut left[compaction.moved_to].layers[layer_idx],
                )
            } else {
                unreachable!("idx < active.len() <= last");
            };

            ctx.stream
                .memcpy_dtod(&src_part.state, &mut dst_part.state)
                .expect("compact slot state copy failed");
            ctx.stream
                .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                .expect("compact slot conv_state copy failed");
        }
        self.graph_state.slot_states[compaction.moved_to].seq_len =
            self.graph_state.slot_states[compaction.moved_from].seq_len;

        match &mut active[compaction.moved_to].backend_state {
            ActiveBackendState::Single { graph_slot_idx, .. } => {
                *graph_slot_idx = compaction.moved_to;
            }
            ActiveBackendState::Tp { .. } => {
                panic!("single-GPU slot compaction received TP active state")
            }
        }
    }
}

impl TpSchedulerBackend {
    fn new(
        model_path: &str,
        device_ordinals: &[usize],
        max_batch: usize,
        max_prefill_tokens: usize,
        prefix_snapshot_bytes: usize,
    ) -> Result<Self> {
        let executor = Qwen35TpExecutor::from_runtime_with_limits_and_prefix(
            model_path,
            false,
            device_ordinals,
            max_batch,
            max_prefill_tokens,
            prefix_snapshot_bytes,
        )?;
        Ok(Self {
            executor,
            next_request_id: 1,
        })
    }

    fn alloc_request_id(&mut self) -> RequestId {
        let id = RequestId::new(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        id
    }

    fn alloc_prefill_state(
        &mut self,
        req: &SchedulerRequest,
    ) -> Result<(PrefillBackendState, usize)> {
        let request_id = self.alloc_request_id();
        let cached_tokens = self.executor.begin_request(
            request_id,
            &req.prompt_tokens,
            req.max_tokens,
            req.lora_adapter.as_deref(),
            !req.echo,
        )?;
        Ok((PrefillBackendState::Tp { request_id }, cached_tokens))
    }

    fn max_batch(&self) -> usize {
        self.executor.max_batch()
    }

    fn page_size(&self) -> usize {
        self.executor.page_size()
    }

    fn capacity_pages_for_requests(&self) -> usize {
        self.executor.capacity_pages_for_requests()
    }

    fn max_position_embeddings(&self) -> usize {
        self.executor.max_position_embeddings()
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.executor.is_stop_token(token)
    }

    fn available_pages(&self) -> usize {
        self.executor.available_pages()
    }

    fn execute_prefill_chunk(
        &mut self,
        chunk: &ScheduledChunk,
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
            anyhow::bail!("TP prefill received single-GPU chunk state");
        };
        let items: Vec<TpPrefillChunkItem> = chunk
            .reqs
            .iter()
            .zip(request_ids)
            .zip(&chunk.windows)
            .zip(&chunk.ends)
            .map(|(((req, request_id), window), end)| {
                TpPrefillChunkItem::new_with_sampling(
                    *request_id,
                    window.clone(),
                    req.logprobs,
                    req.params,
                    *end == req.prompt_tokens.len(),
                )
            })
            .collect();
        let result = self
            .executor
            .execute_prefill_chunks_with_seed(&items, sample_seed)?;
        align_prefill_results(chunk, &result)
    }

    fn execute_decode(
        &mut self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        let items: Vec<TpDecodeStepItem> = active
            .iter()
            .map(|req| {
                let ActiveBackendState::Tp { request_id } = &req.backend_state else {
                    anyhow::bail!("TP decode received single-GPU active state");
                };
                Ok(TpDecodeStepItem::new(
                    *request_id,
                    req.last_token,
                    req.logprobs,
                    req.params,
                ))
            })
            .collect::<Result<_>>()?;
        let result = self.executor.execute_decode_items(&items, sample_seed)?;
        align_decode_results(active, &result)
    }

    fn drop_request(&mut self, request_id: RequestId) {
        if let Err(err) = self.executor.drop_request(request_id) {
            warn!(
                "failed to drop Qwen3.5 TP worker request {}: {err}",
                request_id.get()
            );
        }
    }
}

impl SchedulerBackend {
    fn max_batch(&self) -> usize {
        match self {
            Self::Single(backend) => backend.max_batch(),
            Self::Tp(backend) => backend.max_batch(),
        }
    }

    fn page_size(&self) -> usize {
        match self {
            Self::Single(backend) => backend.page_size(),
            Self::Tp(backend) => backend.page_size(),
        }
    }

    fn available_pages(
        &self,
        active: &[ActiveRequest35],
        prefilling: &[PrefillingRequest35],
    ) -> usize {
        match self {
            Self::Single(backend) => backend.available_pages(),
            Self::Tp(backend) => {
                let _ = (active, prefilling);
                backend.available_pages()
            }
        }
    }

    fn capacity_pages_for_requests(&self) -> usize {
        match self {
            Self::Single(backend) => backend.capacity_pages_for_requests(),
            Self::Tp(backend) => backend.capacity_pages_for_requests(),
        }
    }

    fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Single(backend) => backend.max_position_embeddings(),
            Self::Tp(backend) => backend.max_position_embeddings(),
        }
    }

    fn alloc_prefill_state(
        &mut self,
        req: &SchedulerRequest,
    ) -> Result<(PrefillBackendState, usize)> {
        match self {
            Self::Single(backend) => backend.alloc_prefill_state(req),
            Self::Tp(backend) => backend.alloc_prefill_state(req),
        }
    }

    fn snapshot_stride(&self) -> Option<usize> {
        match self {
            Self::Single(backend) if backend.kv_cache.enabled() => {
                Some(crate::prefix_cache::SNAPSHOT_STRIDE_TOKENS)
            }
            Self::Tp(backend) if backend.executor.prefix_cache_enabled() => {
                Some(crate::prefix_cache::SNAPSHOT_STRIDE_TOKENS)
            }
            Self::Single(_) | Self::Tp(_) => None,
        }
    }

    fn is_tp(&self) -> bool {
        matches!(self, Self::Tp(_))
    }

    fn is_stop_token(&self, token: u32) -> bool {
        match self {
            Self::Single(backend) => backend.is_stop_token(token),
            Self::Tp(backend) => backend.is_stop_token(token),
        }
    }

    fn log_prefix_cache_stats(&self) {
        match self {
            Self::Single(backend) => backend.log_prefix_cache_stats(),
            Self::Tp(backend) => backend.executor.log_prefix_cache_stats(),
        }
    }
}

fn align_prefill_results(
    chunk: &ScheduledChunk,
    result: &PrefillResult,
) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("align_prefill_results requires TP chunk state");
    };
    let mut tokens = vec![0u32; chunk.reqs.len()];
    let mut logprobs = vec![None; chunk.reqs.len()];
    for PrefillRequestResult {
        request_id,
        first_token,
        first_token_logprob,
    } in &result.requests
    {
        let idx = request_ids
            .iter()
            .position(|id| id == request_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP prefill returned unknown request id {}",
                    request_id.get()
                )
            })?;
        tokens[idx] = *first_token;
        logprobs[idx].clone_from(first_token_logprob);
    }
    Ok((tokens, logprobs))
}

fn align_decode_results(
    active: &[ActiveRequest35],
    result: &DecodeResult,
) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
    anyhow::ensure!(
        active.len() == result.requests.len(),
        "Qwen3.5 TP decode result row count mismatch: active={}, result={}",
        active.len(),
        result.requests.len()
    );
    let mut tokens = Vec::with_capacity(active.len());
    let mut logprobs = Vec::with_capacity(active.len());
    for (
        active_req,
        DecodeRequestResult {
            request_id,
            token,
            logprob,
        },
    ) in active.iter().zip(&result.requests)
    {
        let ActiveBackendState::Tp {
            request_id: expected,
        } = &active_req.backend_state
        else {
            anyhow::bail!("align_decode_results requires TP active state");
        };
        anyhow::ensure!(
            *expected == *request_id,
            "Qwen3.5 TP decode result request id mismatch: expected {}, got {}",
            expected.get(),
            request_id.get()
        );
        tokens.push(*token);
        logprobs.push(logprob.clone());
    }
    Ok((tokens, logprobs))
}

fn servable_len(max_context: usize, max_pages: usize, page_size: usize) -> u32 {
    max_context
        .min(max_pages.saturating_mul(page_size))
        .try_into()
        .unwrap_or(u32::MAX)
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

fn bind_model_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 scheduler thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 scheduler thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    model.tune_decode_gemm_algos()?;
    Ok(CublasThreadGuard)
}

// ── Main loop ───────────────────────────────────────────────────────────

fn publish_load(
    load_tx: &watch::Sender<LoadSnapshot>,
    backend: &SchedulerBackend,
    active: &[ActiveRequest35],
    prefilling: &[PrefillingRequest35],
    num_waiting_reqs: usize,
) {
    let kv_total_blocks = backend.capacity_pages_for_requests() as u64;
    load_tx.send_replace(LoadSnapshot {
        kv_used_blocks: kv_total_blocks
            .saturating_sub(backend.available_pages(active, prefilling) as u64),
        kv_total_blocks,
        num_running_reqs: (active.len() + prefilling.len()) as u64,
        num_waiting_reqs: num_waiting_reqs as u64,
    });
}

#[allow(clippy::needless_pass_by_value)]
fn scheduler_loop(
    mut backend: SchedulerBackend,
    mut submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    seed: u64,
    prefill_budget: usize,
    scheduler_policy: Qwen35SchedulerPolicy,
    load_tx: watch::Sender<LoadSnapshot>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest35> = Vec::new();
    let mut deferred: Vec<SchedulerRequest> = Vec::new();
    let mut prefilling: Vec<PrefillingRequest35> = Vec::new();
    let max_batch = backend.max_batch();

    info!("scheduler ready (max_batch={})", max_batch);

    loop {
        // Publish the settled state between scheduler steps. If the prior step
        // retired its final requests, their KV pages have already returned via
        // RAII, so this snapshot reaches idle before the channel blocks below.
        publish_load(&load_tx, &backend, &active, &prefilling, deferred.len());

        // 1. Drain all pending requests (deferred from last iteration + channel)
        let mut pending = std::mem::take(&mut deferred);
        while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
            pending.push(req);
        }

        // 2. Nothing in flight (no decode, no in-progress prefill) and nothing
        //    pending → block until a request arrives.
        if active.is_empty() && prefilling.is_empty() && pending.is_empty() {
            if let Some((req, _kv_prefix)) = submit_rx.blocking_recv() {
                pending.push(req);
            } else {
                info!("scheduler: all handles dropped, exiting");
                backend.log_prefix_cache_stats();
                return;
            }
            while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
                pending.push(req);
            }
        }

        // 3. Admit new prompts. In-flight prefills reserve their promotion slot
        //    and future KV growth, so shrink the slot/page budgets accordingly
        let active_budget: Vec<ActiveKvBudget> = active
            .iter()
            .map(|req| ActiveKvBudget {
                prompt_len: req.prompt_len,
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let page_size = backend.page_size();
        let prefilling_budget: Vec<PrefillKvBudget> = prefilling
            .iter()
            .map(|p| PrefillKvBudget {
                current_tokens: p.cursor,
                prompt_len: p.req.prompt_tokens.len(),
                max_tokens: p.req.max_tokens,
            })
            .collect();
        let page_budget = backend
            .available_pages(&active, &prefilling)
            .saturating_sub(prefilling_future_pages(&prefilling_budget, page_size));
        let decode_batching_slot = max_batch.saturating_sub(prefilling.len());
        // Keep admission's protocol-level prompt + max_tokens limit identical
        // to the max_model_len advertised by the handle. The content-hashed
        // pool reserves one padding block, so its physical cap can be below
        // the model's configured context length.
        let max_context_tokens = servable_len(
            backend.max_position_embeddings(),
            backend.capacity_pages_for_requests(),
            page_size,
        ) as usize;
        let admission = admit_pending_requests(
            pending,
            &active_budget,
            decode_batching_slot,
            page_size,
            page_budget,
            // The block pool includes the CUDA Graph padding page reserved at
            // construction, so a real request can use at most the remaining pages.
            backend.capacity_pages_for_requests(),
            max_context_tokens,
            |req| req.prompt_tokens.len(),
            |req| req.max_tokens,
        );
        for (rejected, reason) in &admission.rejected {
            send_rejection(rejected, *reason);
        }

        // 4. Move freshly admitted prompts into the chunked-prefill queue.
        for req in admission.pending {
            debug!(
                "request admitted: request_id={:?} prompt_len={} max_tokens={}",
                req.request_id,
                req.prompt_tokens.len(),
                req.max_tokens
            );
            match backend.alloc_prefill_state(&req) {
                Ok((backend_state, cached_tokens)) => {
                    let scheduled_at_unix_s = unix_now_s();
                    if req
                        .token_tx
                        .send(TokenEvent::Scheduled {
                            queued_at_unix_s: req.queued_at_unix_s.unwrap_or(scheduled_at_unix_s),
                            scheduled_at_unix_s,
                            prompt_tokens: req.prompt_tokens.len(),
                            cached_tokens,
                        })
                        .is_err()
                    {
                        backend.drop_prefill_state(backend_state);
                        continue;
                    }
                    prefilling.push(PrefillingRequest35 {
                        backend_state,
                        cursor: cached_tokens,
                        step_chunk: 0,
                        req,
                    });
                }
                Err(e) => {
                    warn!("failed to allocate recurrent state for new request: {e}");
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: e.to_string(),
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
            }
        }

        deferred = admission.deferred;

        // 5. Choose this tick's prefill budget, take that chunk off the front of
        //    the queue, then dispatch by plan. Auto can return 0 for a short
        //    decode-priority tick; the next iteration reconsiders the same FIFO
        //    prefill without reordering it.
        let active_decode: Vec<ActiveDecodeState> = active
            .iter()
            .map(|req| ActiveDecodeState {
                generated_count: req.generated_count,
                max_tokens: req.max_tokens,
            })
            .collect();
        let prefill_queue: Vec<PrefillQueueState> = prefilling
            .iter()
            .map(|req| PrefillQueueState {
                remaining_tokens: req.req.prompt_tokens.len().saturating_sub(req.cursor),
            })
            .collect();
        let step_prefill_budget = choose_prefill_budget(
            scheduler_policy,
            prefill_budget,
            &active_decode,
            &prefill_queue,
        );
        let scheduled = take_prefill_chunks(
            &mut prefilling,
            step_prefill_budget,
            backend.snapshot_stride(),
        );
        // ITL diagnostics (#470): capture the *actual* prefill-chunk token count
        // and the frozen decode width for this step before the plan consumes the
        // scheduled set. Off unless PEGAINFER_ITL_DEBUG is set.
        let itl_debug = itl_debug_enabled();
        let itl_prefill_tokens: usize = scheduled.iter().map(|p| p.step_chunk).sum();
        let itl_prefill_reqs = scheduled.len();
        let itl_decode_n = active.len();
        let plan = if backend.is_tp() {
            build_eager_only_plan(!active.is_empty(), scheduled)
        } else {
            plan::build_next_plan(!active.is_empty(), scheduled)
        };
        if let Some(plan) = plan {
            let itl_plan_kind = match &plan {
                ExecutionPlan::Unified { .. } => "unified",
                ExecutionPlan::Prefill { .. } => "prefill",
                ExecutionPlan::Decode => "decode",
            };
            let itl_step_start = itl_debug.then(Instant::now);
            match plan {
                ExecutionPlan::Unified { pending } => unified_step_sched(
                    &mut backend,
                    &mut active,
                    pending,
                    &mut prefilling,
                    &mut rng,
                ),
                ExecutionPlan::Prefill { pending } => prefill_batch(
                    &mut backend,
                    &mut active,
                    pending,
                    &mut prefilling,
                    &mut rng,
                ),
                ExecutionPlan::Decode => {
                    decode_step(&mut backend, &mut active, &mut rng);
                }
            }
            if let Some(step_start) = itl_step_start {
                // A `unified`/`prefill` step with prefill_tok>0 is the only kind
                // that freezes active decodes behind real prefill work; a
                // `decode` step (prefill_tok=0) is a genuine steady gap. dur_us
                // is the CPU wall-time of the step, i.e. the per-step stall the
                // active decodes actually eat this tick.
                let dur_us = step_start.elapsed().as_micros();
                let epoch_us = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_micros());
                info!(
                    "ITL_STEP mono_us={} epoch_us={} plan={} prefill_tok={} prefill_reqs={} decode_n={} dur_us={}",
                    itl_debug_mono_us(),
                    epoch_us,
                    itl_plan_kind,
                    itl_prefill_tokens,
                    itl_prefill_reqs,
                    itl_decode_n,
                    dur_us
                );
            }
        }
    }
}

fn build_eager_only_plan<T>(have_active: bool, pending: Vec<T>) -> Option<ExecutionPlan<T>> {
    if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

fn send_rejection(req: &SchedulerRequest, reason: RejectReason) {
    let message = match reason {
        RejectReason::ContextLength { limit } => format!(
            "request exceeds this model's maximum context length of {limit} tokens: requested {} (prompt={} + max_tokens={})",
            req.prompt_tokens.len().saturating_add(req.max_tokens),
            req.prompt_tokens.len(),
            req.max_tokens
        ),
        RejectReason::KvBudget => {
            let max_request_tokens = max_kv_tokens(req.prompt_tokens.len(), req.max_tokens);
            format!(
                "request requires more KV pages than this model instance can provide: prompt_tokens={}, max_request_tokens={max_request_tokens}",
                req.prompt_tokens.len()
            )
        }
    };
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message,
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    });
}

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
) {
    let mut chunk = ScheduledChunk::from(scheduled);
    let sample_seed = rand::RngExt::random(rng);
    let (tokens, logprobs_vec) = match backend {
        SchedulerBackend::Single(single) => {
            // Scope the borrows of `chunk` to the executor call so the error path can
            // move `chunk` into `fail_chunk`.
            let logits = match single.batch_prefill_logits(&mut chunk) {
                Ok(v) => v,
                Err(e) => {
                    warn!("batch prefill failed: {e}");
                    fail_chunk(single, chunk, &e.to_string());
                    return;
                }
            };
            let sampled = match single.sample_prefill_logits(&chunk.reqs, &logits, rng) {
                Ok(v) => v,
                Err(e) => {
                    warn!("prefill sampling failed: {e}");
                    if let ScheduledChunkBackendState::Single { kvs, .. } = &mut chunk.backend_state
                    {
                        revert_scheduled_requests(&single.kv_cache, kvs);
                    }
                    fail_chunk(single, chunk, &e.to_string());
                    return;
                }
            };
            if let Err(e) = single.apply_prefill(&mut chunk, &sampled.0) {
                warn!("prefill KV/snapshot commit failed: {e}");
                fail_chunk(single, chunk, &e.to_string());
                return;
            }
            sampled
        }
        SchedulerBackend::Tp(tp) => match tp.execute_prefill_chunk(&chunk, sample_seed) {
            Ok(v) => v,
            Err(e) => {
                warn!("TP prefill chunk failed: {e}");
                fail_chunk(backend, chunk, &e.to_string());
                return;
            }
        },
    };

    promote_or_requeue(backend, active, prefilling, chunk, &tokens, &logprobs_vec);
}

// ── Unified step (prefill chunk + decode in one forward pass) ──────────────

fn unified_step_sched(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    scheduled: Vec<PrefillingRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    rng: &mut StdRng,
) {
    let SchedulerBackend::Single(backend) = backend else {
        let chunk = ScheduledChunk::from(scheduled);
        let message = "Qwen3.5 TP Phase 1 does not support unified prefill+decode steps";
        warn!("{message}");
        for req in active.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Error {
                message: message.to_string(),
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
        }
        fail_chunk(backend, chunk, message);
        return;
    };
    let mut chunk = ScheduledChunk::from(scheduled);
    // Scope the borrows of `chunk` / `active` to the executor call so the error
    // and decode-processing paths can use them afterwards.
    let result = backend.unified_step(&mut chunk, active);
    let output = match result {
        Ok(v) => v,
        Err(e) => {
            warn!("unified step failed: {e}");
            let message = e.to_string();
            backend.fail_active(active, &message);
            fail_chunk(backend, chunk, &message);
            return;
        }
    };

    let prefill_logits = output
        .prefill_logits
        .as_ref()
        .expect("scheduled prefill chunk must return prefill logits");
    let decode_sampled = if output.decoded {
        match backend.sample_decode_logits(active, rng) {
            Ok(sampled) => Some(sampled),
            Err(e) => {
                warn!("unified decode sampling failed: {e}");
                revert_scheduled_requests(
                    &backend.kv_cache,
                    active.iter_mut().filter_map(active_request_kv),
                );
                if let ScheduledChunkBackendState::Single { kvs, .. } = &mut chunk.backend_state {
                    revert_scheduled_requests(&backend.kv_cache, kvs);
                }
                let message = e.to_string();
                backend.fail_active(active, &message);
                fail_chunk(backend, chunk, &message);
                return;
            }
        }
    } else {
        None
    };
    let (tokens, logprobs_vec) =
        match backend.sample_prefill_logits(&chunk.reqs, prefill_logits, rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("unified prefill sampling failed: {e}");
                revert_scheduled_requests(
                    &backend.kv_cache,
                    active.iter_mut().filter_map(active_request_kv),
                );
                if let ScheduledChunkBackendState::Single { kvs, .. } = &mut chunk.backend_state {
                    revert_scheduled_requests(&backend.kv_cache, kvs);
                }
                backend.fail_active(active, &e.to_string());
                fail_chunk(backend, chunk, &e.to_string());
                return;
            }
        };

    if let Some((decode_tokens, _)) = &decode_sampled
        && let Err(e) = backend.apply_decode(active, decode_tokens)
    {
        warn!("unified decode KV apply failed: {e}");
        backend.fail_active(active, &e.to_string());
        fail_chunk(backend, chunk, &e.to_string());
        return;
    }
    if let Err(e) = backend.apply_prefill(&mut chunk, &tokens) {
        warn!("unified prefill KV/snapshot commit failed: {e}");
        backend.fail_active(active, &e.to_string());
        fail_chunk(backend, chunk, &e.to_string());
        return;
    }

    // Decode commits and dispatches first; retirements free graph slots that
    // the newly-prefilled requests can then occupy densely.
    if let Some((decode_tokens, decode_logprobs)) = decode_sampled {
        dispatch_decode_tokens(backend, active, &decode_tokens, &decode_logprobs);
    }
    promote_or_requeue(backend, active, prefilling, chunk, &tokens, &logprobs_vec);
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    backend: &mut SchedulerBackend,
    active: &mut Vec<ActiveRequest35>,
    rng: &mut StdRng,
) {
    let sample_seed = rand::RngExt::random(rng);
    let (tokens, logprobs_vec) = match backend {
        SchedulerBackend::Single(single) => {
            if let Err(e) = single.decode_graph(active) {
                warn!("batch_decode_graph error: {e}");
                let message = e.to_string();
                single.fail_active(active, &message);
                return;
            }
            // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits)
            match single.sample_decode_logits(active, rng) {
                Ok(v) => v,
                Err(e) => {
                    warn!("decode sampling/logprobs error: {e}");
                    revert_scheduled_requests(
                        &single.kv_cache,
                        active.iter_mut().filter_map(active_request_kv),
                    );
                    let message = e.to_string();
                    single.fail_active(active, &message);
                    return;
                }
            }
        }
        SchedulerBackend::Tp(tp) => match tp.execute_decode(active, sample_seed) {
            Ok(v) => v,
            Err(e) => {
                warn!("TP eager decode error: {e}");
                let message = e.to_string();
                for req in active.drain(..) {
                    let state = req.backend_state;
                    if let ActiveBackendState::Tp { request_id } = state {
                        tp.drop_request(request_id);
                    }
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                return;
            }
        },
    };

    if let SchedulerBackend::Single(single) = backend {
        if let Err(e) = single.apply_decode(active, &tokens) {
            warn!("decode KV apply failed: {e}");
            let message = e.to_string();
            single.fail_active(active, &message);
            return;
        }
    }
    dispatch_decode_tokens(backend, active, &tokens, &logprobs_vec);
}

/// Dispatch sampled decode tokens: send events, check EOS/limits, retire finished.
///
/// `tokens` and `logprobs` are indexed by original position in `active`.
/// Retirements collected first, then compacted in reverse order.
fn dispatch_decode_tokens(
    backend: &mut impl DecodeDispatchBackend,
    active: &mut Vec<ActiveRequest35>,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
) {
    let n = active.len();
    let mut to_retire = Vec::new();

    for i in 0..n {
        let token = tokens[i];
        let logprob = logprobs[i].clone();
        let req = &mut active[i];
        req.generated_count += 1;

        let is_eos = !req.params.ignore_eos && backend.is_stop_token(token);
        let at_limit = req.generated_count >= req.max_tokens;

        if is_eos {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                req.prompt_len,
                req.generated_count,
                FinishReason::Stop
            );
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            to_retire.push(i);
        } else if at_limit {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                req.prompt_len,
                req.generated_count,
                FinishReason::Length
            );
            let _ = req.token_tx.send(TokenEvent::Token { id: token, logprob });
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            to_retire.push(i);
        } else if req
            .token_tx
            .send(TokenEvent::Token { id: token, logprob })
            .is_err()
        {
            debug!(
                "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                req.request_id, req.generated_count
            );
            to_retire.push(i);
        } else {
            req.last_token = token;
        }
    }

    // Remove in reverse order so compact_slot indices stay valid
    for &i in to_retire.iter().rev() {
        backend.retire_request(active, i);
    }
}

trait DecodeDispatchBackend {
    fn is_stop_token(&self, token: u32) -> bool;
    fn retire_request(&mut self, active: &mut Vec<ActiveRequest35>, idx: usize);
}

impl DecodeDispatchBackend for SingleGpuBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn retire_request(&mut self, active: &mut Vec<ActiveRequest35>, idx: usize) {
        compact_single_slot(self, active, idx);
    }
}

impl DecodeDispatchBackend for SchedulerBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn retire_request(&mut self, active: &mut Vec<ActiveRequest35>, idx: usize) {
        match self {
            SchedulerBackend::Single(backend) => compact_single_slot(backend, active, idx),
            SchedulerBackend::Tp(backend) => {
                let removed = active.swap_remove(idx);
                if let ActiveBackendState::Tp { request_id } = removed.backend_state {
                    backend.drop_request(request_id);
                }
            }
        }
    }
}

/// Remove single-GPU request at `idx` via swap_remove and compact graph slots.
///
/// After swap_remove, the element that was at `active.len()-1` (before remove)
/// now sits at `idx`. Its graph slot must be copied into the vacated slot so
/// that slots 0..active.len() remain dense.
fn compact_single_slot(
    backend: &mut SingleGpuBackend,
    active: &mut Vec<ActiveRequest35>,
    idx: usize,
) {
    let compaction = compaction_after_retire(active.len(), idx);
    let mut removed = active.swap_remove(idx);
    if let ActiveBackendState::Single { kv, .. } = &mut removed.backend_state {
        if let Err(error) = backend.kv_cache.release_request(kv) {
            warn!("failed to release Qwen3.5 request KV: {error}");
        }
    }

    if let Some(compaction) = compaction {
        backend.compact_slot(active, compaction);
    }
}

// ── Chunked-prefill helpers ────────────────────────────────────────────────

/// Step's scheduled prefill set
struct ScheduledChunk {
    reqs: Vec<SchedulerRequest>,
    backend_state: ScheduledChunkBackendState,
    /// Prompt cursor after this step's chunk
    ends: Vec<usize>,
    /// This step's chunked token slice per request
    windows: Vec<Vec<u32>>,
}

enum ScheduledChunkBackendState {
    Single {
        kvs: Vec<RequestKv>,
        recs: Vec<RecurrentState>,
    },
    Tp {
        request_ids: Vec<RequestId>,
    },
}

impl From<Vec<PrefillingRequest35>> for ScheduledChunk {
    fn from(scheduled: Vec<PrefillingRequest35>) -> Self {
        let n = scheduled.len();
        let is_tp = scheduled
            .first()
            .is_some_and(|p| matches!(p.backend_state, PrefillBackendState::Tp { .. }));
        let mut chunk = ScheduledChunk {
            reqs: Vec::with_capacity(n),
            backend_state: if is_tp {
                ScheduledChunkBackendState::Tp {
                    request_ids: Vec::with_capacity(n),
                }
            } else {
                ScheduledChunkBackendState::Single {
                    kvs: Vec::with_capacity(n),
                    recs: Vec::with_capacity(n),
                }
            },
            ends: Vec::with_capacity(n),
            windows: Vec::with_capacity(n),
        };
        for p in scheduled {
            let end = p.cursor + p.step_chunk;
            chunk
                .windows
                .push(p.req.prompt_tokens[p.cursor..end].to_vec());
            chunk.ends.push(end);
            chunk.reqs.push(p.req);
            match (&mut chunk.backend_state, p.backend_state) {
                (
                    ScheduledChunkBackendState::Single { kvs, recs },
                    PrefillBackendState::Single { kv, rec },
                ) => {
                    kvs.push(*kv);
                    recs.push(rec);
                }
                (
                    ScheduledChunkBackendState::Tp { request_ids },
                    PrefillBackendState::Tp { request_id },
                ) => request_ids.push(request_id),
                _ => unreachable!("mixed Qwen3.5 scheduler backend states in one chunk"),
            }
        }
        chunk
    }
}

/// Pull this step's prefill set off the FRONT of `prefilling`, capping the
/// step's total forwarded prompt tokens at `prefill_budget`.
fn take_prefill_chunks(
    prefilling: &mut Vec<PrefillingRequest35>,
    prefill_budget: usize,
    snapshot_stride: Option<usize>,
) -> Vec<PrefillingRequest35> {
    let remaining: Vec<usize> = prefilling
        .iter()
        .map(|p| p.req.prompt_tokens.len() - p.cursor)
        .collect();
    let chunks = plan_prefill_chunks(&remaining, prefill_budget);
    let mut scheduled: Vec<PrefillingRequest35> = prefilling.drain(0..chunks.len()).collect();
    for (p, chunk) in scheduled.iter_mut().zip(&chunks) {
        p.step_chunk = clamp_prefill_chunk(p.cursor, *chunk, snapshot_stride);
    }
    scheduled
}

fn clamp_prefill_chunk(cursor: usize, chunk: usize, snapshot_stride: Option<usize>) -> usize {
    snapshot_stride.map_or(chunk, |stride| {
        debug_assert!(stride > 0);
        chunk.min(stride - cursor % stride)
    })
}

/// Report a forward/sampling failure to every request in the failed chunk.
fn fail_chunk(backend: &mut impl PrefillPromoteBackend, chunk: ScheduledChunk, message: &str) {
    let states = split_scheduled_backend_state(chunk.backend_state);
    for (req, state) in chunk.reqs.into_iter().zip(states) {
        let _ = req.token_tx.send(TokenEvent::Error {
            message: message.to_string(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        backend.drop_prefill_state(state);
    }
}

/// For each request in the just-prefilled chunk: if its prompt is now exhausted,
/// sample its first token, emit events, and move it into the decode batch;
/// otherwise re-queue it (with an advanced cursor) at the FRONT of `prefilling`.
/// `tokens` / `logprobs` are indexed by request order in `chunk`.
fn promote_or_requeue(
    backend: &mut impl PrefillPromoteBackend,
    active: &mut Vec<ActiveRequest35>,
    prefilling: &mut Vec<PrefillingRequest35>,
    chunk: ScheduledChunk,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
) {
    let ScheduledChunk {
        reqs,
        backend_state,
        ends,
        ..
    } = chunk;
    let mut still_prefilling: Vec<PrefillingRequest35> = Vec::new();
    let backend_states = split_scheduled_backend_state(backend_state);

    for (i, ((req, backend_state), end)) in
        reqs.into_iter().zip(backend_states).zip(ends).enumerate()
    {
        // Not finished: re-queue with the advanced cursor
        if end < req.prompt_tokens.len() {
            still_prefilling.push(PrefillingRequest35 {
                req,
                backend_state,
                cursor: end,
                step_chunk: 0,
            });
            continue;
        }

        let prompt_len = req.prompt_tokens.len();
        let first_token = tokens[i];
        let logprob = logprobs[i].clone();

        if req.echo {
            let echo_logprobs = vec![None; req.prompt_tokens.len()];
            let _ = req.token_tx.send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }

        if !req.params.ignore_eos && backend.is_stop_token(first_token) {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                prompt_len,
                0,
                FinishReason::Stop
            );
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            backend.drop_prefill_state(backend_state);
            continue;
        }

        if req
            .token_tx
            .send(TokenEvent::Token {
                id: first_token,
                logprob,
            })
            .is_err()
        {
            debug!(
                "request dropped: client disconnected: request_id={:?} tokens_generated={}",
                req.request_id, 0
            );
            backend.drop_prefill_state(backend_state);
            continue;
        }

        if req.max_tokens <= 1 {
            debug!(
                "request finished: request_id={:?} prompt_tokens={} completion_tokens={} finish_reason={:?}",
                req.request_id,
                prompt_len,
                1,
                FinishReason::Length
            );
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            backend.drop_prefill_state(backend_state);
            continue;
        }

        let active_backend_state = backend.promote_prefill_state(active.len(), backend_state);
        active.push(ActiveRequest35 {
            request_id: req.request_id,
            token_tx: req.token_tx,
            backend_state: active_backend_state,
            last_token: first_token,
            generated_count: 1,
            max_tokens: req.max_tokens,
            prompt_len,
            params: req.params,
            logprobs: req.logprobs,
        });
    }

    prefilling.splice(0..0, still_prefilling);
}

trait PrefillPromoteBackend {
    fn is_stop_token(&self, token: u32) -> bool;
    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState;
    fn drop_prefill_state(&mut self, state: PrefillBackendState);
}

impl PrefillPromoteBackend for SingleGpuBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        let PrefillBackendState::Single { kv, rec } = state else {
            panic!("single-GPU promotion received TP prefill state");
        };
        let slot_idx = slot_for_new_request(active_len, self.max_batch())
            .expect("admission must reserve a graph slot");
        self.copy_recurrent_to_slot(&rec, slot_idx)
            .expect("copy recurrent state to slot failed");
        ActiveBackendState::Single {
            kv,
            graph_slot_idx: slot_idx,
        }
    }

    fn drop_prefill_state(&mut self, state: PrefillBackendState) {
        let PrefillBackendState::Single { mut kv, .. } = state else {
            panic!("single-GPU drop received TP prefill state");
        };
        if let Err(error) = self.kv_cache.release_request(&mut kv) {
            warn!("failed to release Qwen3.5 request KV: {error}");
        }
    }
}

impl PrefillPromoteBackend for SchedulerBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        match (self, state) {
            (SchedulerBackend::Single(single), PrefillBackendState::Single { kv, rec }) => {
                let slot_idx = slot_for_new_request(active_len, single.max_batch())
                    .expect("admission must reserve a graph slot");
                single
                    .copy_recurrent_to_slot(&rec, slot_idx)
                    .expect("copy recurrent state to slot failed");
                ActiveBackendState::Single {
                    kv,
                    graph_slot_idx: slot_idx,
                }
            }
            (SchedulerBackend::Tp(_), PrefillBackendState::Tp { request_id }) => {
                ActiveBackendState::Tp { request_id }
            }
            _ => panic!("mismatched Qwen3.5 scheduler backend state during promotion"),
        }
    }

    fn drop_prefill_state(&mut self, state: PrefillBackendState) {
        match (self, state) {
            (SchedulerBackend::Single(backend), PrefillBackendState::Single { mut kv, .. }) => {
                if let Err(error) = backend.kv_cache.release_request(&mut kv) {
                    warn!("failed to release Qwen3.5 request KV: {error}");
                }
            }
            (SchedulerBackend::Tp(backend), PrefillBackendState::Tp { request_id }) => {
                backend.drop_request(request_id);
            }
            _ => panic!("mismatched Qwen3.5 scheduler backend state during drop"),
        }
    }
}

fn split_scheduled_backend_state(
    backend_state: ScheduledChunkBackendState,
) -> Vec<PrefillBackendState> {
    match backend_state {
        ScheduledChunkBackendState::Single { kvs, recs } => kvs
            .into_iter()
            .zip(recs)
            .map(|(kv, rec)| PrefillBackendState::Single {
                kv: Box::new(kv),
                rec,
            })
            .collect(),
        ScheduledChunkBackendState::Tp { request_ids } => request_ids
            .into_iter()
            .map(|request_id| PrefillBackendState::Tp { request_id })
            .collect(),
    }
}

#[cfg(test)]
mod tests;
