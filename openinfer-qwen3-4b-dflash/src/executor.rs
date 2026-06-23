use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use half::bf16;
use openinfer_core::tensor::{DeviceContext, HiddenStates};

use crate::batch_buffers::DFlashBatchBuffers;
use crate::batch_forward::{DFlashBatchInput, DFlashHostBatchInput, copy_hidden};
use crate::forward::{DFlashDraftCache, DFlashTargetHidden};
use crate::weights::DFlashDraftModel;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DFlashRequestId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DFlashCacheMode {
    NoCache,
    DraftCache,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DFlashBatchKey {
    pub q_len: usize,
    pub ctx_len: usize,
    pub past_len: usize,
    pub cache_mode: DFlashCacheMode,
}

pub struct DFlashDraftRequest {
    pub request_id: DFlashRequestId,
    pub noise_embedding: HiddenStates,
    pub target_hidden: HiddenStates,
    pub position_ids: Vec<i32>,
    pub cache_mode: DFlashCacheMode,
}

pub struct DFlashDraftHostRequest {
    pub request_id: DFlashRequestId,
    pub noise_embedding: Vec<bf16>,
    pub target_hidden: Vec<bf16>,
    pub position_ids: Vec<i32>,
    pub q_len: usize,
    pub ctx_len: usize,
    pub cache_mode: DFlashCacheMode,
}

pub struct DFlashDraftResponse {
    pub request_id: DFlashRequestId,
    pub output: HiddenStates,
    pub cache_seq_len: usize,
    pub batch_size: usize,
    pub elapsed: Duration,
}

pub struct DFlashDraftHostResponse {
    pub request_id: DFlashRequestId,
    pub output: Vec<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub cache_seq_len: usize,
    pub batch_size: usize,
    pub elapsed: Duration,
}

pub struct DFlashDraftBatchResponse {
    pub request_ids: Vec<DFlashRequestId>,
    pub output: HiddenStates,
    pub cache_seq_lens: Vec<usize>,
    pub batch_size: usize,
    pub q_len: usize,
    pub elapsed: Duration,
}

pub struct DFlashDraftBatchView<'a> {
    pub request_ids: Vec<DFlashRequestId>,
    pub output: &'a HiddenStates,
    pub cache_seq_lens: Vec<usize>,
    pub batch_size: usize,
    pub q_len: usize,
    pub elapsed: Duration,
}

pub struct DFlashExecutorOptions {
    pub max_batch_size: usize,
    pub max_step_context_len: usize,
    /// Largest draft length (`q_len`) the executor must serve. Batch buffers
    /// are sized once for `max_batch_size × max_q_len`, so every shape at or
    /// below it reuses the same allocation (mirrors Qwen3's `BatchDecodeBuffers`).
    pub max_q_len: usize,
    pub max_seq_len: usize,
    /// Upper bound on resident draft caches. Each `DraftCache` request creates
    /// a per-request `DFlashDraftCache` (full `ForwardBuffers` + per-layer past
    /// K/V); without a cap they accumulate forever and leak GPU memory.
    /// Admission fails closed when this is exceeded — callers must `drop_cache`
    /// a retired request before submitting a new one. Mirrors Qwen3's per-
    /// request block accounting under the fixed `KvCacheManager` pool.
    pub max_caches: usize,
}

impl Default for DFlashExecutorOptions {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_step_context_len: 16,
            max_q_len: 16,
            max_seq_len: 4096,
            max_caches: 64,
        }
    }
}

pub struct DFlashExecutor {
    model: DFlashDraftModel,
    options: DFlashExecutorOptions,
    /// Single-instance batch buffer, sized for the worst case
    /// (`max_batch_size × max_q_len × max_step_context_len`). Each forward
    /// narrows the active shape via `set_active_shape` instead of reallocating.
    buffers: DFlashBatchBuffers,
    caches: HashMap<DFlashRequestId, DFlashDraftCache>,
}

impl DFlashExecutor {
    pub fn load(
        model_path: &Path,
        device_ordinal: usize,
        options: DFlashExecutorOptions,
    ) -> Result<Self> {
        let model = DFlashDraftModel::load(model_path, device_ordinal)?;
        let buffers = model.create_batch_buffers(
            options.max_batch_size,
            options.max_q_len,
            options.max_step_context_len,
        )?;
        Ok(Self {
            model,
            options,
            buffers,
            caches: HashMap::new(),
        })
    }

    pub fn model(&self) -> &DFlashDraftModel {
        &self.model
    }

    pub fn max_batch_size(&self) -> usize {
        self.options.max_batch_size
    }

    pub fn batch_key(&self, req: &DFlashDraftRequest) -> Result<DFlashBatchKey> {
        let target = DFlashTargetHidden {
            concatenated: &req.target_hidden,
        };
        let (q_len, ctx_len) =
            self.model
                .validate_forward_inputs(&req.noise_embedding, &target, &req.position_ids)?;
        let past_len = self
            .caches
            .get(&req.request_id)
            .map(DFlashDraftCache::seq_len)
            .unwrap_or(0);
        Ok(DFlashBatchKey {
            q_len,
            ctx_len,
            past_len,
            cache_mode: req.cache_mode,
        })
    }

    pub fn host_batch_key(&self, req: &DFlashDraftHostRequest) -> Result<DFlashBatchKey> {
        let config = self.model.config();
        anyhow::ensure!(
            req.noise_embedding.len() == req.q_len * config.hidden_size,
            "noise_embedding len {} != q_len * hidden_size {}",
            req.noise_embedding.len(),
            req.q_len * config.hidden_size
        );
        anyhow::ensure!(
            req.target_hidden.len()
                == req.ctx_len * config.hidden_size * config.target_layer_count(),
            "target_hidden len {} != ctx_len * target_layer_count * hidden_size {}",
            req.target_hidden.len(),
            req.ctx_len * config.hidden_size * config.target_layer_count()
        );
        anyhow::ensure!(
            req.position_ids.len() == req.ctx_len + req.q_len,
            "position_ids len {} != ctx_len + q_len {}",
            req.position_ids.len(),
            req.ctx_len + req.q_len
        );
        let past_len = self
            .caches
            .get(&req.request_id)
            .map(DFlashDraftCache::seq_len)
            .unwrap_or(0);
        Ok(DFlashBatchKey {
            q_len: req.q_len,
            ctx_len: req.ctx_len,
            past_len,
            cache_mode: req.cache_mode,
        })
    }

    pub fn execute_batch(
        &mut self,
        requests: Vec<DFlashDraftRequest>,
    ) -> Result<Vec<DFlashDraftResponse>> {
        let batch = self.execute_batch_compact(requests)?;
        self.split_compact_response(batch)
    }

    pub fn execute_host_batch_compact(
        &mut self,
        requests: Vec<DFlashDraftHostRequest>,
    ) -> Result<DFlashDraftBatchResponse> {
        anyhow::ensure!(!requests.is_empty(), "DFlash host executor batch is empty");
        anyhow::ensure!(
            requests.len() <= self.options.max_batch_size,
            "DFlash host executor batch size {} exceeds max_batch_size {}",
            requests.len(),
            self.options.max_batch_size
        );
        let key = self.host_batch_key(&requests[0])?;
        for req in &requests[1..] {
            let req_key = self.host_batch_key(req)?;
            anyhow::ensure!(
                req_key == key,
                "DFlash host executor requires exact-shape batch: first={key:?}, got={req_key:?}"
            );
        }
        if key.cache_mode == DFlashCacheMode::DraftCache {
            return self.execute_cached_host_requests_serial_compact(requests, key);
        }
        anyhow::ensure!(
            key.q_len <= self.options.max_q_len,
            "DFlash host q_len {} exceeds executor max_q_len {}",
            key.q_len,
            self.options.max_q_len
        );
        anyhow::ensure!(
            key.ctx_len <= self.options.max_step_context_len,
            "DFlash host ctx_len {} exceeds executor max_step_context_len {}",
            key.ctx_len,
            self.options.max_step_context_len
        );
        let started = Instant::now();
        let batch_size = requests.len();
        let request_ids = requests
            .iter()
            .map(|request| request.request_id)
            .collect::<Vec<_>>();
        let inputs = requests
            .iter()
            .map(|req| DFlashHostBatchInput {
                noise_embedding: &req.noise_embedding,
                target_hidden: &req.target_hidden,
                position_ids: &req.position_ids,
            })
            .collect::<Vec<_>>();
        let batch_output = self.model.forward_host_batch(&inputs, &mut self.buffers)?;
        self.model.device_context().sync()?;
        let elapsed = started.elapsed();
        // forward returns a borrow into self.buffers; materialize an owned copy
        // so the next batch can reuse the buffer without aliasing the response.
        let output = clone_batch_output(self.model.device_context(), batch_output)?;
        Ok(DFlashDraftBatchResponse {
            request_ids,
            output,
            cache_seq_lens: vec![0; batch_size],
            batch_size,
            q_len: key.q_len,
            elapsed,
        })
    }

    pub fn execute_host_batch(
        &mut self,
        requests: Vec<DFlashDraftHostRequest>,
    ) -> Result<Vec<DFlashDraftResponse>> {
        let batch = self.execute_host_batch_compact(requests)?;
        self.split_compact_response(batch)
    }

    pub fn execute_host_batch_host(
        &mut self,
        requests: Vec<DFlashDraftHostRequest>,
    ) -> Result<Vec<DFlashDraftHostResponse>> {
        let batch = self.execute_host_batch_compact(requests)?;
        self.split_compact_host_response(batch)
    }

    pub fn execute_host_batch_view(
        &mut self,
        requests: Vec<DFlashDraftHostRequest>,
    ) -> Result<DFlashDraftBatchView<'_>> {
        anyhow::ensure!(!requests.is_empty(), "DFlash host executor batch is empty");
        anyhow::ensure!(
            requests.len() <= self.options.max_batch_size,
            "DFlash host executor batch size {} exceeds max_batch_size {}",
            requests.len(),
            self.options.max_batch_size
        );
        let key = self.host_batch_key(&requests[0])?;
        for req in &requests[1..] {
            let req_key = self.host_batch_key(req)?;
            anyhow::ensure!(
                req_key == key,
                "DFlash host executor requires exact-shape batch: first={key:?}, got={req_key:?}"
            );
        }
        anyhow::ensure!(
            key.cache_mode == DFlashCacheMode::NoCache,
            "borrowed host batch view currently supports only NoCache mode"
        );
        anyhow::ensure!(
            key.q_len <= self.options.max_q_len,
            "DFlash host q_len {} exceeds executor max_q_len {}",
            key.q_len,
            self.options.max_q_len
        );
        anyhow::ensure!(
            key.ctx_len <= self.options.max_step_context_len,
            "DFlash host ctx_len {} exceeds executor max_step_context_len {}",
            key.ctx_len,
            self.options.max_step_context_len
        );
        let started = Instant::now();
        let batch_size = requests.len();
        let request_ids = requests
            .iter()
            .map(|request| request.request_id)
            .collect::<Vec<_>>();
        let inputs = requests
            .iter()
            .map(|req| DFlashHostBatchInput {
                noise_embedding: &req.noise_embedding,
                target_hidden: &req.target_hidden,
                position_ids: &req.position_ids,
            })
            .collect::<Vec<_>>();
        let output = self.model.forward_host_batch(&inputs, &mut self.buffers)?;
        self.model.device_context().sync()?;
        Ok(DFlashDraftBatchView {
            request_ids,
            output,
            cache_seq_lens: vec![0; batch_size],
            batch_size,
            q_len: key.q_len,
            elapsed: started.elapsed(),
        })
    }

    pub fn execute_batch_compact(
        &mut self,
        requests: Vec<DFlashDraftRequest>,
    ) -> Result<DFlashDraftBatchResponse> {
        anyhow::ensure!(!requests.is_empty(), "DFlash executor batch is empty");
        anyhow::ensure!(
            requests.len() <= self.options.max_batch_size,
            "DFlash executor batch size {} exceeds max_batch_size {}",
            requests.len(),
            self.options.max_batch_size
        );
        let key = self.batch_key(&requests[0])?;
        for req in &requests[1..] {
            let req_key = self.batch_key(req)?;
            anyhow::ensure!(
                req_key == key,
                "DFlash executor requires exact-shape batch: first={key:?}, got={req_key:?}"
            );
        }
        match key.cache_mode {
            DFlashCacheMode::NoCache => self.execute_uncached_batch_compact(requests, key),
            DFlashCacheMode::DraftCache => {
                self.execute_cached_requests_serial_compact(requests, key)
            }
        }
    }

    pub fn reset_cache(&mut self, request_id: DFlashRequestId) -> Result<()> {
        let Some(cache) = self.caches.get_mut(&request_id) else {
            anyhow::bail!("unknown DFlash cache request_id {:?}", request_id);
        };
        cache.reset();
        Ok(())
    }

    pub fn crop_cache(&mut self, request_id: DFlashRequestId, seq_len: usize) -> Result<()> {
        let Some(cache) = self.caches.get_mut(&request_id) else {
            anyhow::bail!("unknown DFlash cache request_id {:?}", request_id);
        };
        cache.crop(seq_len)?;
        Ok(())
    }

    pub fn cache_seq_len(&self, request_id: DFlashRequestId) -> Result<usize> {
        self.caches
            .get(&request_id)
            .map(DFlashDraftCache::seq_len)
            .ok_or_else(|| anyhow::anyhow!("unknown DFlash cache request_id {:?}", request_id))
    }

    /// Release a request's draft cache. Mirrors Qwen3's `drop_request`
    /// (`openinfer-qwen3-4b/src/executor.rs`): remove the entry and let RAII
    /// drop the GPU buffers. Idempotent — a missing cache is not an error, so
    /// callers can retire a request from any lifecycle state.
    pub fn drop_cache(&mut self, request_id: DFlashRequestId) -> Result<()> {
        self.caches.remove(&request_id);
        Ok(())
    }

    /// Resident cache count, for admission diagnostics.
    pub fn cache_count(&self) -> usize {
        self.caches.len()
    }

    /// Ensure a draft cache exists for `request_id`, enforcing the
    /// `max_caches` cap. Existing caches are reused (a re-submitted request
    /// keeps its past state). Over-cap admission fails closed. Returns without
    /// borrowing the cache so callers can then use disjoint `&self.model` and
    /// `&mut self.caches` borrows in the same scope (NLL split borrow).
    fn ensure_cache_entry(
        &mut self,
        request_id: DFlashRequestId,
        key: &DFlashBatchKey,
    ) -> Result<()> {
        if !self.caches.contains_key(&request_id) {
            anyhow::ensure!(
                self.caches.len() < self.options.max_caches,
                "DFlash cache pool full: {} resident caches, max_caches={}; drop_cache a retired request before submitting a new one",
                self.caches.len(),
                self.options.max_caches,
            );
            let cache = self.model.create_draft_cache(
                key.q_len,
                self.options.max_step_context_len,
                self.options.max_seq_len,
            )?;
            self.caches.insert(request_id, cache);
        }
        Ok(())
    }

    fn execute_uncached_batch_compact(
        &mut self,
        requests: Vec<DFlashDraftRequest>,
        key: DFlashBatchKey,
    ) -> Result<DFlashDraftBatchResponse> {
        anyhow::ensure!(
            key.q_len <= self.options.max_q_len,
            "DFlash q_len {} exceeds executor max_q_len {}",
            key.q_len,
            self.options.max_q_len
        );
        anyhow::ensure!(
            key.ctx_len <= self.options.max_step_context_len,
            "DFlash ctx_len {} exceeds executor max_step_context_len {}",
            key.ctx_len,
            self.options.max_step_context_len
        );
        let started = Instant::now();
        let batch_size = requests.len();
        let request_ids = requests
            .iter()
            .map(|request| request.request_id)
            .collect::<Vec<_>>();
        let inputs = requests
            .iter()
            .map(|req| DFlashBatchInput {
                noise_embedding: &req.noise_embedding,
                target_hidden: DFlashTargetHidden {
                    concatenated: &req.target_hidden,
                },
                position_ids: &req.position_ids,
            })
            .collect::<Vec<_>>();
        let batch_output = self.model.forward_batch(&inputs, &mut self.buffers)?;
        self.model.device_context().sync()?;
        let elapsed = started.elapsed();
        let output = clone_batch_output(self.model.device_context(), batch_output)?;
        Ok(DFlashDraftBatchResponse {
            request_ids,
            output,
            cache_seq_lens: vec![0; batch_size],
            batch_size,
            q_len: key.q_len,
            elapsed,
        })
    }

    fn execute_cached_requests_serial_compact(
        &mut self,
        requests: Vec<DFlashDraftRequest>,
        key: DFlashBatchKey,
    ) -> Result<DFlashDraftBatchResponse> {
        let started = Instant::now();
        let batch_size = requests.len();
        let mut request_ids = Vec::with_capacity(batch_size);
        let mut cache_seq_lens = Vec::with_capacity(batch_size);
        let mut output = HiddenStates::zeros(
            self.model.device_context(),
            self.model.config().hidden_size,
            batch_size * key.q_len,
        )?;
        for (i, req) in requests.into_iter().enumerate() {
            self.ensure_cache_entry(req.request_id, &key)?;
            let cache = self.caches.get_mut(&req.request_id).expect("cache exists");
            self.model.prepare_step_context(
                DFlashTargetHidden {
                    concatenated: &req.target_hidden,
                },
                &req.position_ids,
                cache,
            )?;
            let out = self.model.forward_with_draft_cache(
                &req.noise_embedding,
                &req.position_ids,
                cache,
            )?;
            self.model.device_context().sync()?;
            copy_hidden(
                self.model.device_context(),
                out,
                0,
                &mut output,
                i * key.q_len,
                self.model.config().hidden_size,
                key.q_len,
            )?;
            request_ids.push(req.request_id);
            cache_seq_lens.push(cache.seq_len());
        }
        Ok(DFlashDraftBatchResponse {
            request_ids,
            output,
            cache_seq_lens,
            batch_size,
            q_len: key.q_len,
            elapsed: started.elapsed(),
        })
    }

    fn execute_cached_host_requests_serial_compact(
        &mut self,
        requests: Vec<DFlashDraftHostRequest>,
        key: DFlashBatchKey,
    ) -> Result<DFlashDraftBatchResponse> {
        let started = Instant::now();
        let batch_size = requests.len();
        let config = self.model.config();
        let hidden = config.hidden_size;
        let target_hidden_dim = config.hidden_size * config.target_layer_count();
        let mut request_ids = Vec::with_capacity(batch_size);
        let mut cache_seq_lens = Vec::with_capacity(batch_size);
        let mut output =
            HiddenStates::zeros(self.model.device_context(), hidden, batch_size * key.q_len)?;
        for (i, req) in requests.into_iter().enumerate() {
            let noise_embedding = HiddenStates {
                data: self
                    .model
                    .device_context()
                    .stream
                    .clone_htod(&req.noise_embedding)?,
                hidden_dim: hidden,
                seq_len: key.q_len,
            };
            let target_hidden = HiddenStates {
                data: self
                    .model
                    .device_context()
                    .stream
                    .clone_htod(&req.target_hidden)?,
                hidden_dim: target_hidden_dim,
                seq_len: key.ctx_len,
            };
            self.ensure_cache_entry(req.request_id, &key)?;
            let cache = self.caches.get_mut(&req.request_id).expect("cache exists");
            self.model.prepare_step_context(
                DFlashTargetHidden {
                    concatenated: &target_hidden,
                },
                &req.position_ids,
                cache,
            )?;
            let out =
                self.model
                    .forward_with_draft_cache(&noise_embedding, &req.position_ids, cache)?;
            self.model.device_context().sync()?;
            copy_hidden(
                self.model.device_context(),
                out,
                0,
                &mut output,
                i * key.q_len,
                hidden,
                key.q_len,
            )?;
            request_ids.push(req.request_id);
            cache_seq_lens.push(cache.seq_len());
        }
        Ok(DFlashDraftBatchResponse {
            request_ids,
            output,
            cache_seq_lens,
            batch_size,
            q_len: key.q_len,
            elapsed: started.elapsed(),
        })
    }

    fn split_compact_response(
        &self,
        batch: DFlashDraftBatchResponse,
    ) -> Result<Vec<DFlashDraftResponse>> {
        let mut responses = Vec::with_capacity(batch.batch_size);
        for i in 0..batch.batch_size {
            let mut output = HiddenStates::zeros(
                self.model.device_context(),
                self.model.config().hidden_size,
                batch.q_len,
            )?;
            copy_hidden(
                self.model.device_context(),
                &batch.output,
                i * batch.q_len,
                &mut output,
                0,
                self.model.config().hidden_size,
                batch.q_len,
            )?;
            responses.push(DFlashDraftResponse {
                request_id: batch.request_ids[i],
                output,
                cache_seq_len: batch.cache_seq_lens[i],
                batch_size: batch.batch_size,
                elapsed: batch.elapsed,
            });
        }
        Ok(responses)
    }

    fn split_compact_host_response(
        &self,
        batch: DFlashDraftBatchResponse,
    ) -> Result<Vec<DFlashDraftHostResponse>> {
        let host = self
            .model
            .device_context()
            .stream
            .clone_dtoh(&batch.output.data)?;
        self.model.device_context().sync()?;
        let row_len = batch.output.hidden_dim * batch.q_len;
        let mut responses = Vec::with_capacity(batch.batch_size);
        for i in 0..batch.batch_size {
            responses.push(DFlashDraftHostResponse {
                request_id: batch.request_ids[i],
                output: host[i * row_len..(i + 1) * row_len].to_vec(),
                hidden_dim: batch.output.hidden_dim,
                seq_len: batch.q_len,
                cache_seq_len: batch.cache_seq_lens[i],
                batch_size: batch.batch_size,
                elapsed: batch.elapsed,
            });
        }
        Ok(responses)
    }
}

/// Materialize an owned snapshot of a batch forward's output (a borrow into
/// the single-instance buffer). One allocation + one device-to-device copy of
/// the active region; the next batch may overwrite the buffer immediately.
fn clone_batch_output(ctx: &DeviceContext, src: &HiddenStates) -> Result<HiddenStates> {
    let mut dst = HiddenStates::zeros(ctx, src.hidden_dim, src.seq_len)?;
    let len = src.hidden_dim * src.seq_len;
    let src_view = src.data.slice(..len);
    let mut dst_view = dst.data.slice_mut(..len);
    ctx.stream.memcpy_dtod(&src_view, &mut dst_view)?;
    Ok(dst)
}
