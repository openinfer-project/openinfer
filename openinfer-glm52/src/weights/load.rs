use std::{
    cell::Cell,
    collections::{BTreeMap, VecDeque},
    path::Path,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use bytesize::ByteSize;
use cudarc::driver::{CudaEvent, CudaSlice};
use log::{debug, error, info};
use memmap2::Mmap;

use super::{Glm52RankGpuContext, Glm52RankLoadBundle, expected_tensor_contract, mmap_file};

// One mmap per event keeps source lifetimes explicit without the large-tail
// regression observed when many shard mappings stay live together.
const MMAP_EVENT_GROUP: usize = 1;
const LIVE_MMAP_GROUPS: usize = 16;

pub(crate) struct Glm52GpuRawTensor {
    pub(crate) bytes: usize,
    pub(crate) _offset: usize,
}

pub(crate) struct Glm52RankGpuWeights {
    pub(crate) tensors: BTreeMap<String, Glm52GpuRawTensor>,
    pub(crate) _slab: CudaSlice<u8>,
    pub(crate) total_bytes: usize,
}

pub(crate) struct Glm52RankLoadOutput {
    pub(crate) weights: Glm52RankGpuWeights,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
}

pub(crate) fn load_rank_weights_to_gpu(
    ctx: &Glm52RankGpuContext,
    model_path: &Path,
    bundle: &Glm52RankLoadBundle,
) -> Result<Glm52RankLoadOutput> {
    ctx.set_current()?;
    let load_started = Instant::now();
    let planned_total_bytes = bundle.planned_total_bytes()?;
    let alloc_started = Instant::now();
    // SAFETY: every byte in the rank-local slab is written exactly once from a
    // validated safetensors view before the slab becomes resident model state.
    let slab = unsafe { ctx.stream().alloc::<u8>(planned_total_bytes) }.with_context(|| {
        format!(
            "failed to allocate GLM5.2 rank {} weight slab of {}",
            bundle.plan.rank,
            ByteSize(planned_total_bytes as u64)
        )
    })?;
    let alloc_secs = alloc_started.elapsed().as_secs_f64();

    let mut weights = Glm52RankGpuWeights {
        tensors: BTreeMap::new(),
        _slab: slab,
        total_bytes: 0,
    };
    let mut loaded_tensor_count = 0usize;
    let mut loaded_total_bytes = 0usize;
    let copy_started = Instant::now();
    let mut slowest_shard: Option<(String, f64)> = None;
    let mut next_offset = 0usize;
    let mut h2d_copies = 0usize;
    let mut mmap_guard = MmapH2dLifetimeGuard::new(ctx, bundle.plan.rank);
    let mut sync_secs = 0.0f64;
    debug!(
        "GLM5.2 rank {} start weight load: tensors={}, shards={}, bytes={}, non_expert={}, experts={:?}",
        bundle.plan.rank,
        bundle.plan.tensor_count,
        bundle.shards.len(),
        ByteSize(planned_total_bytes as u64),
        bundle.plan.loads_non_expert,
        bundle.plan.expert_range
    );

    for shard in &bundle.shards {
        let path = model_path.join(&shard.shard);
        let shard_started = Instant::now();
        let mmap = CurrentShardMmap::new(mmap_file(&path)?, ctx, bundle.plan.rank);
        let safetensors = safetensors::SafeTensors::deserialize(mmap.as_ref())
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        let mmap_base = mmap.as_ref().as_ptr() as usize;
        let mut pending = Vec::with_capacity(shard.tensors.len());
        for spec in &shard.tensors {
            let view = safetensors
                .tensor(&spec.name)
                .with_context(|| format!("missing tensor {} in {}", spec.name, path.display()))?;
            let contract = expected_tensor_contract(&spec.name)?;
            ensure!(
                view.dtype() == contract.dtype,
                "GLM5.2 tensor {} dtype mismatch: got {:?}, expected {:?}",
                spec.name,
                view.dtype(),
                contract.dtype
            );
            ensure!(
                view.shape() == contract.shape.as_slice(),
                "GLM5.2 tensor {} shape mismatch: got {:?}, expected {:?}",
                spec.name,
                view.shape(),
                contract.shape
            );
            let bytes = view.data().len();
            ensure!(
                bytes == contract.byte_len()?,
                "GLM5.2 tensor {} byte mismatch: got {}, expected {}",
                spec.name,
                bytes,
                contract.byte_len()?
            );
            let data_start = view
                .data()
                .as_ptr()
                .addr()
                .checked_sub(mmap_base)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM5.2 tensor {} data pointer is outside {}",
                        spec.name,
                        path.display()
                    )
                })?;
            let data_end = data_start.checked_add(bytes).ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM5.2 tensor {} source byte range overflow in {}",
                    spec.name,
                    path.display()
                )
            })?;
            ensure!(
                data_end <= mmap.as_ref().len(),
                "GLM5.2 tensor {} source range [{data_start}..{data_end}) exceeds shard bytes {}",
                spec.name,
                mmap.as_ref().len()
            );
            pending.push(PendingTensor {
                name: spec.name.as_str(),
                data_start,
                bytes,
            });
        }

        pending.sort_by_key(|tensor| tensor.data_start);
        let mut shard_tensors = Vec::with_capacity(pending.len());
        for pending_tensor in pending {
            let offset = next_offset;
            next_offset = next_offset
                .checked_add(pending_tensor.bytes)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM5.2 rank {} slab offset overflow while loading {}",
                        bundle.plan.rank,
                        pending_tensor.name
                    )
                })?;
            ensure!(
                next_offset <= planned_total_bytes,
                "GLM5.2 rank {} slab overflow while loading {}: next_offset={}, slab_bytes={}",
                bundle.plan.rank,
                pending_tensor.name,
                next_offset,
                planned_total_bytes
            );
            shard_tensors.push(PlacedTensor {
                name: pending_tensor.name,
                data_start: pending_tensor.data_start,
                data_end: pending_tensor.data_start + pending_tensor.bytes,
                dst_start: offset,
                dst_end: next_offset,
                bytes: pending_tensor.bytes,
            });
        }

        let mut run_start = 0usize;
        while run_start < shard_tensors.len() {
            let first = &shard_tensors[run_start];
            let mut run_end = run_start + 1;
            let mut src_end = first.data_end;
            let mut dst_end = first.dst_end;
            while run_end < shard_tensors.len() {
                let current = &shard_tensors[run_end];
                if current.data_start != src_end || current.dst_start != dst_end {
                    break;
                }
                src_end = current.data_end;
                dst_end = current.dst_end;
                run_end += 1;
            }

            {
                let src = &mmap.as_ref()[first.data_start..src_end];
                let mut dst = weights._slab.slice_mut(first.dst_start..dst_end);
                mmap.mark_stream_work();
                ctx.stream().memcpy_htod(src, &mut dst).with_context(|| {
                    format!(
                        "failed to copy GLM5.2 rank {} shard {} byte range [{}..{}) to GPU",
                        bundle.plan.rank, shard.shard, first.data_start, src_end
                    )
                })?;
            }
            h2d_copies += 1;

            for tensor in &shard_tensors[run_start..run_end] {
                let raw = Glm52GpuRawTensor {
                    bytes: tensor.bytes,
                    _offset: tensor.dst_start,
                };
                weights.total_bytes += raw.bytes;
                loaded_total_bytes += raw.bytes;
                loaded_tensor_count += 1;
                ensure!(
                    weights
                        .tensors
                        .insert(tensor.name.to_owned(), raw)
                        .is_none(),
                    "duplicate GLM5.2 tensor {} in rank {} load plan",
                    tensor.name,
                    bundle.plan.rank,
                );
            }
            run_start = run_end;
        }
        drop(safetensors);
        sync_secs += mmap_guard.push_completed_shard(mmap.into_mmap())?;
        let shard_secs = shard_started.elapsed().as_secs_f64();
        match &slowest_shard {
            Some((_, slowest_secs)) if *slowest_secs >= shard_secs => {}
            _ => slowest_shard = Some((shard.shard.clone(), shard_secs)),
        }
    }

    ensure!(
        loaded_tensor_count == bundle.plan.tensor_count,
        "GLM5.2 rank {} loaded {loaded_tensor_count} tensors but load plan has {}",
        bundle.plan.rank,
        bundle.plan.tensor_count
    );
    ensure!(
        next_offset == planned_total_bytes,
        "GLM5.2 rank {} loaded {} bytes but load plan has {} bytes",
        bundle.plan.rank,
        next_offset,
        planned_total_bytes
    );
    sync_secs += mmap_guard.drain()?;
    let copy_secs = (copy_started.elapsed().as_secs_f64() - sync_secs).max(0.0);
    let sync_started = Instant::now();
    ctx.sync().with_context(|| {
        format!(
            "failed to finish GLM5.2 rank {} H2D tensor copies",
            bundle.plan.rank
        )
    })?;
    sync_secs += sync_started.elapsed().as_secs_f64();

    let (slowest_shard, slowest_secs) = slowest_shard.unwrap_or_else(|| ("none".to_owned(), 0.0));
    info!(
        "GLM5.2 rank {} weight load profile: total={:.2}s, alloc={:.2}s, mmap_deser_copy={:.2}s, sync={:.2}s, tensors={}, h2d_copies={}, bytes={}, slowest_shard={} {:.2}s",
        bundle.plan.rank,
        load_started.elapsed().as_secs_f64(),
        alloc_secs,
        copy_secs,
        sync_secs,
        loaded_tensor_count,
        h2d_copies,
        ByteSize(loaded_total_bytes as u64),
        slowest_shard,
        slowest_secs
    );

    Ok(Glm52RankLoadOutput {
        weights,
        loaded_tensor_count,
        loaded_total_bytes,
    })
}

struct PendingTensor<'a> {
    name: &'a str,
    data_start: usize,
    bytes: usize,
}

struct PlacedTensor<'a> {
    name: &'a str,
    data_start: usize,
    data_end: usize,
    dst_start: usize,
    dst_end: usize,
    bytes: usize,
}

struct LiveMmapGroup {
    _mmaps: Vec<Mmap>,
    event: CudaEvent,
}

struct CurrentShardMmap<'a> {
    mmap: Option<Mmap>,
    ctx: &'a Glm52RankGpuContext,
    rank: usize,
    stream_work: Cell<bool>,
}

impl<'a> CurrentShardMmap<'a> {
    fn new(mmap: Mmap, ctx: &'a Glm52RankGpuContext, rank: usize) -> Self {
        Self {
            mmap: Some(mmap),
            ctx,
            rank,
            stream_work: Cell::new(false),
        }
    }

    fn as_ref(&self) -> &Mmap {
        self.mmap.as_ref().expect("current shard mmap is present")
    }

    fn mark_stream_work(&self) {
        self.stream_work.set(true);
    }

    fn into_mmap(mut self) -> Mmap {
        self.stream_work.set(false);
        self.mmap.take().expect("current shard mmap is present")
    }
}

impl Drop for CurrentShardMmap<'_> {
    fn drop(&mut self) {
        if self.stream_work.get() {
            if let Err(error) = self.ctx.set_current().and_then(|_| self.ctx.sync()) {
                error!(
                    "failed to synchronize GLM5.2 rank {} before dropping current H2D mmap: {error:#}",
                    self.rank
                );
            }
        }
    }
}

struct MmapH2dLifetimeGuard<'a> {
    ctx: &'a Glm52RankGpuContext,
    rank: usize,
    pending_mmaps: Vec<Mmap>,
    live_mmap_groups: VecDeque<LiveMmapGroup>,
}

impl<'a> MmapH2dLifetimeGuard<'a> {
    fn new(ctx: &'a Glm52RankGpuContext, rank: usize) -> Self {
        Self {
            ctx,
            rank,
            pending_mmaps: Vec::with_capacity(MMAP_EVENT_GROUP),
            live_mmap_groups: VecDeque::with_capacity(LIVE_MMAP_GROUPS),
        }
    }

    fn push_completed_shard(&mut self, mmap: Mmap) -> Result<f64> {
        self.pending_mmaps.push(mmap);
        let mut sync_secs = 0.0;
        if self.pending_mmaps.len() == MMAP_EVENT_GROUP {
            self.record_pending_group()?;
        }
        if self.live_mmap_groups.len() >= LIVE_MMAP_GROUPS {
            sync_secs += self.wait_oldest_group()?;
        }
        Ok(sync_secs)
    }

    fn drain(&mut self) -> Result<f64> {
        let mut sync_secs = 0.0;
        if !self.pending_mmaps.is_empty() {
            self.record_pending_group()?;
        }
        while !self.live_mmap_groups.is_empty() {
            sync_secs += self.wait_oldest_group()?;
        }
        Ok(sync_secs)
    }

    fn record_pending_group(&mut self) -> Result<()> {
        let event = self.ctx.stream().record_event(None).with_context(|| {
            format!(
                "failed to record GLM5.2 rank {} H2D mmap group event",
                self.rank
            )
        })?;
        let mut mmaps = Vec::with_capacity(MMAP_EVENT_GROUP);
        std::mem::swap(&mut self.pending_mmaps, &mut mmaps);
        self.live_mmap_groups.push_back(LiveMmapGroup {
            _mmaps: mmaps,
            event,
        });
        Ok(())
    }

    fn wait_oldest_group(&mut self) -> Result<f64> {
        let Some(live) = self.live_mmap_groups.front() else {
            return Ok(0.0);
        };
        let started = Instant::now();
        live.event.synchronize().with_context(|| {
            format!(
                "failed to wait for GLM5.2 rank {} H2D mmap event",
                self.rank
            )
        })?;
        self.live_mmap_groups.pop_front();
        Ok(started.elapsed().as_secs_f64())
    }
}

impl Drop for MmapH2dLifetimeGuard<'_> {
    fn drop(&mut self) {
        if self.pending_mmaps.is_empty() && self.live_mmap_groups.is_empty() {
            return;
        }
        if let Err(error) = self.ctx.set_current().and_then(|_| self.ctx.sync()) {
            error!(
                "failed to synchronize GLM5.2 rank {} before dropping H2D mmap guard: {error:#}",
                self.rank
            );
        }
    }
}
