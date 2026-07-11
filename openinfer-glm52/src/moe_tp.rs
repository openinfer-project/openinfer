//! GLM5.2 tensor-parallel MoE: per-rank slices of all 257 experts
//! (shared folded at bank index 256) + the phase-kernel chain state (LL
//! packet buffers, cross-rank pointer exchange, scratch arena). Replicated
//! activations: every rank passes all 8 rows and receives all 8 reduced
//! rows back, bit-identical. Design:
//! `docs/models/glm52/moe-tp8-low-latency.md`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_TP_BANK_EXPERTS, GLM52_TP_HIDDEN, GLM52_TP_MAX_RANKS, GLM52_TP_TOKENS,
    GLM52_TP_UNION_MAX, Glm52MoeTpBuffers, Glm52TpLlBuffer, Glm52TpTopology,
    glm52_moe_tp_epoch_advance, glm52_moe_tp_layer_launch, glm52_moe_tp_max_blocks,
    glm52_tp_ar_buffer_bytes, glm52_tp_ar_chunk_packets, glm52_tp_ar_launch,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::config::{GLM52_EXPERT_INTERMEDIATE as INTERMEDIATE, GLM52_HIDDEN};
use crate::moe_decode::{EXPERTS, QUANT_GROUP, W2_K, W2_N, W2_SCALE_COLS, W2_SCALE_ROWS};
use crate::weights::{Glm52WeightManifest, expected_tensor_contract, mmap_file, retype_owned};

const H: usize = GLM52_TP_HIDDEN;
const RANKS: usize = GLM52_TP_MAX_RANKS;

// The replicated shape assumes the scheduler's largest bucket IS the
// kernel's row count: a bigger bucket would pass every >= buffer ensure
// while the kernel silently computes only 8 rows (stale mlp_out on the
// rest).
const _: () =
    assert!(crate::model::GLM52_MAX_BATCH_PER_RANK == openinfer_kernels::ops::GLM52_TP_TOKENS);
const BANK: usize = GLM52_TP_BANK_EXPERTS;
#[cfg(test)]
const SLICE_ROWS: usize = Glm52TpTopology::Tp8.slice_rows();
#[cfg(test)]
const SLICE_I: usize = Glm52TpTopology::Tp8.slice_i();

/// One pilot layer's TP slice bank: this rank's intermediate rows of all 257
/// experts, in the layout the cooperative kernel consumes.
pub(crate) struct Glm52MoeTpSliceBank {
    pub(crate) tp_ranks: usize,
    pub(crate) slice_i: usize,
    pub(crate) slice_rows: usize,
    pub(crate) w13: CudaSlice<u8>,        // fp8 [257, 512, 6144]
    pub(crate) w13_scale: CudaSlice<f32>, // f32 [257, 4, 48]
    pub(crate) w2: CudaSlice<u8>,         // fp8 [257, 6144, 256]
    pub(crate) w2_scale: CudaSlice<f32>,  // f32 [257, 48, 2]
}

/// Slice one expert's checkpoint tensors into the rank-r staging bank.
/// `bank_idx` is the destination expert slot (routed id, or 256 for shared).
struct SliceStaging {
    rank: usize,
    tp_ranks: usize,
    slice_i: usize,
    slice_rows: usize,
    w13: Vec<u8>,
    w13_scale: Vec<u8>,
    w2: Vec<u8>,
    w2_scale: Vec<u8>,
}

/// Projection kind for one checkpoint tensor loaded into a TP slice bank.
#[derive(Clone, Copy)]
enum SliceKind {
    Gate,
    Up,
    Down,
    GateScale,
    UpScale,
    DownScale,
}

impl SliceStaging {
    fn new(rank: usize, tp_ranks: usize) -> Result<Self> {
        ensure!(
            tp_ranks > 0 && INTERMEDIATE.is_multiple_of(tp_ranks),
            "GLM5.2 TP slice count {tp_ranks} must divide expert intermediate {INTERMEDIATE}"
        );
        ensure!(
            rank < tp_ranks,
            "GLM5.2 TP slice rank {rank} out of range for {tp_ranks} ranks"
        );
        let slice_i = INTERMEDIATE / tp_ranks;
        let slice_rows = 2 * slice_i;
        ensure!(
            slice_i.is_multiple_of(QUANT_GROUP) && slice_rows.is_multiple_of(QUANT_GROUP),
            "GLM5.2 TP slice geometry must align to FP8 quant group {QUANT_GROUP}: \
             slice_i={slice_i}, slice_rows={slice_rows}"
        );
        Ok(Self {
            rank,
            tp_ranks,
            slice_i,
            slice_rows,
            w13: vec![0u8; BANK * slice_rows * H],
            w13_scale: vec![0u8; BANK * (slice_rows / QUANT_GROUP) * (H / QUANT_GROUP) * 4],
            w2: vec![0u8; BANK * H * slice_i],
            w2_scale: vec![0u8; BANK * (H / QUANT_GROUP) * (slice_i / QUANT_GROUP) * 4],
        })
    }

    /// gate/up [2048, 6144]: rows r*256..(r+1)*256 land at slice rows 0..256
    /// (gate) / 256..512 (up) — one contiguous copy each.
    fn put_w13_weight(&mut self, bank_idx: usize, is_up: bool, src: &[u8]) {
        debug_assert_eq!(src.len(), INTERMEDIATE * H);
        let rows = self.slice_i; // rows per projection per rank
        let src_off = self.rank * rows * H;
        let dst_off = bank_idx * self.slice_rows * H + if is_up { self.slice_i * H } else { 0 };
        self.w13[dst_off..dst_off + rows * H].copy_from_slice(&src[src_off..src_off + rows * H]);
    }

    /// gate/up scale f32 [16, 48]: row blocks 2r..2r+2 land at slice blocks
    /// 0..2 (gate) / 2..4 (up).
    fn put_w13_scale(&mut self, bank_idx: usize, is_up: bool, src: &[u8]) {
        debug_assert_eq!(
            src.len(),
            (INTERMEDIATE / QUANT_GROUP) * (H / QUANT_GROUP) * 4
        );
        let row_bytes = (H / QUANT_GROUP) * 4; // 48 f32
        let blocks = self.slice_i / QUANT_GROUP;
        let src_off = self.rank * blocks * row_bytes;
        let dst_off = (bank_idx * (self.slice_rows / QUANT_GROUP) + if is_up { blocks } else { 0 })
            * row_bytes;
        self.w13_scale[dst_off..dst_off + blocks * row_bytes]
            .copy_from_slice(&src[src_off..src_off + blocks * row_bytes]);
    }

    /// down [6144, 2048]: columns r*256..(r+1)*256 of every row — strided
    /// gather into [6144, 256].
    fn put_w2_weight(&mut self, bank_idx: usize, src: &[u8]) {
        debug_assert_eq!(src.len(), W2_N * W2_K);
        let dst_base = bank_idx * H * self.slice_i;
        let src_col = self.rank * self.slice_i;
        for row in 0..H {
            let dst = dst_base + row * self.slice_i;
            let src_off = row * W2_K + src_col;
            self.w2[dst..dst + self.slice_i].copy_from_slice(&src[src_off..src_off + self.slice_i]);
        }
    }

    /// down scale f32 [48, 16]: column blocks 2r..2r+2 of every row block.
    fn put_w2_scale(&mut self, bank_idx: usize, src: &[u8]) {
        debug_assert_eq!(src.len(), W2_SCALE_ROWS * W2_SCALE_COLS * 4);
        let blocks = self.slice_i / QUANT_GROUP;
        let dst_base = bank_idx * W2_SCALE_ROWS * blocks * 4;
        let src_col = self.rank * blocks * 4;
        for row in 0..W2_SCALE_ROWS {
            let dst = dst_base + row * blocks * 4;
            let src_off = row * W2_SCALE_COLS * 4 + src_col;
            self.w2_scale[dst..dst + blocks * 4]
                .copy_from_slice(&src[src_off..src_off + blocks * 4]);
        }
    }

    fn upload(self, ctx: &DeviceContext) -> Result<Glm52MoeTpSliceBank> {
        let htod = |host: &[u8]| -> Result<CudaSlice<u8>> {
            // SAFETY: fully written by the memcpy below before use.
            let mut dst = unsafe { ctx.stream.alloc::<u8>(host.len()) }?;
            ctx.stream.memcpy_htod(host, &mut dst)?;
            Ok(dst)
        };
        Ok(Glm52MoeTpSliceBank {
            tp_ranks: self.tp_ranks,
            slice_i: self.slice_i,
            slice_rows: self.slice_rows,
            w13: htod(&self.w13)?,
            w13_scale: retype_owned::<f32>(&ctx.stream, htod(&self.w13_scale)?)?,
            w2: htod(&self.w2)?,
            w2_scale: retype_owned::<f32>(&ctx.stream, htod(&self.w2_scale)?)?,
        })
    }
}

/// Second-pass load of one pilot layer's TP8 slice bank for `rank`: re-reads
/// every expert's fp8 tensors (plus the shared expert) from the checkpoint
/// shards and gathers this rank's 1/8-I slice host-side, then uploads.
#[allow(dead_code)] // Used by GPU oracle gates; production uses topology-aware load_tp_slice_layer.
pub(crate) fn load_tp8_slice_layer(
    ctx: &DeviceContext,
    model_path: &Path,
    manifest: &Glm52WeightManifest,
    rank: usize,
    layer: usize,
) -> Result<Glm52MoeTpSliceBank> {
    load_tp_slice_layer(ctx, model_path, manifest, rank, RANKS, layer)
}

/// Second-pass load of one tensor-replicated MoE slice bank for `rank`.
/// TP8 uses 1/8-intermediate slices; TP4 uses 1/4-intermediate slices.
pub(crate) fn load_tp_slice_layer(
    ctx: &DeviceContext,
    model_path: &Path,
    manifest: &Glm52WeightManifest,
    rank: usize,
    tp_ranks: usize,
    layer: usize,
) -> Result<Glm52MoeTpSliceBank> {
    ensure!(
        rank < tp_ranks,
        "TP rank {rank} out of range for {tp_ranks}"
    );
    // (name, bank_idx, projection kind) for all 257 experts x 6 tensors.
    let mut wanted: Vec<(String, usize, SliceKind)> = Vec::with_capacity(BANK * 6);
    let prefix = format!("model.layers.{layer}.mlp");
    let push_expert =
        |stem: String, bank_idx: usize, wanted: &mut Vec<(String, usize, SliceKind)>| {
            wanted.push((
                format!("{stem}.gate_proj.weight"),
                bank_idx,
                SliceKind::Gate,
            ));
            wanted.push((format!("{stem}.up_proj.weight"), bank_idx, SliceKind::Up));
            wanted.push((
                format!("{stem}.down_proj.weight"),
                bank_idx,
                SliceKind::Down,
            ));
            wanted.push((
                format!("{stem}.gate_proj.weight_scale_inv"),
                bank_idx,
                SliceKind::GateScale,
            ));
            wanted.push((
                format!("{stem}.up_proj.weight_scale_inv"),
                bank_idx,
                SliceKind::UpScale,
            ));
            wanted.push((
                format!("{stem}.down_proj.weight_scale_inv"),
                bank_idx,
                SliceKind::DownScale,
            ));
        };
    for expert in 0..BANK - 1 {
        push_expert(format!("{prefix}.experts.{expert}"), expert, &mut wanted);
    }
    push_expert(format!("{prefix}.shared_experts"), BANK - 1, &mut wanted);

    let mut by_shard: BTreeMap<String, Vec<(String, usize, SliceKind)>> = BTreeMap::new();
    for (name, bank_idx, kind) in wanted {
        let shard = manifest.shard_for(&name)?.to_owned();
        by_shard
            .entry(shard)
            .or_default()
            .push((name, bank_idx, kind));
    }

    let mut staging = SliceStaging::new(rank, tp_ranks)?;
    let mut placed = 0usize;
    for (shard, tensors) in by_shard {
        let path = model_path.join(&shard);
        let mmap = mmap_file(&path)?;
        let safetensors = safetensors::SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to deserialize {}", path.display()))?;
        for (name, bank_idx, kind) in tensors {
            let view = safetensors
                .tensor(&name)
                .with_context(|| format!("missing tensor {name} in {}", path.display()))?;
            let contract = expected_tensor_contract(&name)?;
            ensure!(
                view.dtype() == contract.dtype && view.shape() == contract.shape.as_slice(),
                "GLM5.2 TP tensor {name} contract mismatch: got {:?} {:?}, expected {:?} {:?}",
                view.dtype(),
                view.shape(),
                contract.dtype,
                contract.shape
            );
            let data = view.data();
            match kind {
                SliceKind::Gate => staging.put_w13_weight(bank_idx, false, data),
                SliceKind::Up => staging.put_w13_weight(bank_idx, true, data),
                SliceKind::Down => staging.put_w2_weight(bank_idx, data),
                SliceKind::GateScale => staging.put_w13_scale(bank_idx, false, data),
                SliceKind::UpScale => staging.put_w13_scale(bank_idx, true, data),
                SliceKind::DownScale => staging.put_w2_scale(bank_idx, data),
            }
            placed += 1;
        }
    }
    ensure!(
        placed == BANK * 6,
        "GLM5.2 TP layer {layer} slice load placed {placed} tensors, expected {}",
        BANK * 6
    );
    staging.upload(ctx)
}

/// One rank's published LL buffer mappings: per-accessor VA tables (indexed
/// by fleet device ordinal) for its MoE all-reduce buffer and its attention
/// (o_proj epilogue) all-reduce buffer.
#[derive(Clone, Copy)]
struct Glm52TpLlVas {
    rs: [u64; RANKS],
    ar: [u64; RANKS],
}

/// Cross-rank rendezvous for LL buffer mappings: every rank publishes its
/// per-accessor VA tables (or its setup failure, as a poison pill) and blocks
/// until the topology's ranks are in. Also owns the shutdown-side rendezvous so no rank
/// unmaps LL buffers a peer's in-flight kernels could still be reading.
pub(crate) struct Glm52TpExchange {
    rank_count: usize,
    slots: Mutex<[Option<Result<Glm52TpLlVas, String>>; RANKS]>,
    all_in: Condvar,
    departed: Mutex<usize>,
    all_out: Condvar,
}

impl Glm52TpExchange {
    pub(crate) fn new(rank_count: usize) -> Self {
        assert!(
            rank_count > 0 && rank_count <= RANKS,
            "TP exchange rank count out of range"
        );
        Self {
            rank_count,
            slots: Mutex::new(std::array::from_fn(|_| None)),
            all_in: Condvar::new(),
            departed: Mutex::new(0),
            all_out: Condvar::new(),
        }
    }

    /// Publish this rank's mappings — or its failure. A failed rank MUST
    /// still publish (the `Err` is the poison pill): otherwise the other 7
    /// ranks would block forever and the launch error would surface as a
    /// silent hang instead of a message.
    fn publish_and_wait(
        &self,
        rank: usize,
        vas: Result<Glm52TpLlVas, String>,
    ) -> Result<[Glm52TpLlVas; RANKS]> {
        let mut slots = self.slots.lock().expect("TP8 exchange poisoned");
        ensure!(
            rank < self.rank_count,
            "TP exchange rank {rank} out of range for {} ranks",
            self.rank_count
        );
        ensure!(
            slots[rank].is_none(),
            "TP exchange rank {rank} published twice"
        );
        slots[rank] = Some(vas);
        self.all_in.notify_all();
        while slots[..self.rank_count].iter().any(Option::is_none) {
            let (guard, timeout) = self
                .all_in
                .wait_timeout(slots, Duration::from_secs(120))
                .expect("TP8 exchange poisoned");
            slots = guard;
            if timeout.timed_out() && slots[..self.rank_count].iter().any(Option::is_none) {
                let missing: Vec<usize> = (0..self.rank_count)
                    .filter(|&r| slots[r].is_none())
                    .collect();
                bail!(
                    "TP LL rendezvous timed out after 120s — rank(s) {missing:?} never \
                     published (worker died before reaching the exchange?)"
                );
            }
        }
        let failed: Vec<String> = slots
            .iter()
            .take(self.rank_count)
            .enumerate()
            .filter_map(|(r, s)| match s {
                Some(Err(err)) => Some(format!("rank {r}: {err}")),
                _ => None,
            })
            .collect();
        ensure!(
            failed.is_empty(),
            "TP LL setup failed on peer rank(s): {}",
            failed.join("; ")
        );
        Ok(std::array::from_fn(|r| {
            if r < self.rank_count {
                *slots[r]
                    .as_ref()
                    .expect("checked above")
                    .as_ref()
                    .expect("failures bailed above")
            } else {
                Glm52TpLlVas {
                    rs: [0; RANKS],
                    ar: [0; RANKS],
                }
            }
        }))
    }

    /// Shutdown-side barrier: the caller must have synchronized its stream
    /// first (its own kernels are retired). Once all 8 ranks arrive, no
    /// kernel anywhere can still touch an LL buffer, so every rank may unmap
    /// in any order. On timeout (a peer died mid-serving and will never
    /// arrive) we log and proceed — the dead peer's device may see an
    /// illegal-address on its already-doomed context, which beats hanging
    /// the whole process shutdown.
    pub(crate) fn teardown_rendezvous(&self, rank: usize) {
        let mut departed = self.departed.lock().expect("TP8 exchange poisoned");
        *departed += 1;
        self.all_out.notify_all();
        while *departed < self.rank_count {
            let (guard, timeout) = self
                .all_out
                .wait_timeout(departed, Duration::from_secs(120))
                .expect("TP8 exchange poisoned");
            departed = guard;
            if timeout.timed_out() && *departed < self.rank_count {
                log::warn!(
                    "GLM5.2 rank {rank} TP teardown rendezvous timed out ({}/{} arrived) \
                     — unmapping anyway; a peer rank likely died",
                    *departed,
                    self.rank_count
                );
                return;
            }
        }
    }
}

/// A rank's complete tensor-replicated MoE runtime: the state plus the
/// per-layer slice banks (keyed by absolute layer index).
pub(crate) struct Glm52MoeTpRank {
    pub(crate) state: Glm52MoeTpState,
    pub(crate) slices: BTreeMap<usize, Glm52MoeTpSliceBank>,
}

impl Glm52MoeTpRank {
    /// This layer's TP pieces: runtime state, LL slot index (the layer's
    /// position among this rank's sliced layers), and slice bank.
    pub(crate) fn layer_bank(
        &mut self,
        layer: usize,
    ) -> Option<(&mut Glm52MoeTpState, usize, &Glm52MoeTpSliceBank)> {
        let slot = self.slices.range(..layer).count();
        let bank = self.slices.get(&layer)?;
        Some((&mut self.state, slot, bank))
    }
}

/// Per-rank tensor-parallel runtime state shared by TP4 and TP8.
pub(crate) struct Glm52MoeTpState {
    topology: Glm52TpTopology,
    rank: usize,
    ar_slots: usize,
    grid_blocks: usize,
    _rs: Glm52TpLlBuffer,
    _ar: Glm52TpLlBuffer,
    rs_local: u64,
    ar_local: u64,
    peer_rs: [u64; RANKS],
    peer_ar: [u64; RANKS],
    epoch_dev: CudaSlice<u64>,
    active_rows_dev: CudaSlice<i32>,
    guidx: CudaSlice<i32>,
    guprob: CudaSlice<f32>,
    gucnt: CudaSlice<i32>,
    gused: CudaSlice<i32>,
    ug: CudaSlice<bf16>,
    cpart: CudaSlice<f32>,
}

impl Glm52MoeTpState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        topology: Glm52TpTopology,
        rank: usize,
        device_ordinal: usize,
        exchange: &Glm52TpExchange,
        slots: usize,
        ar_slots: usize,
    ) -> Result<Self> {
        let ranks = topology.ranks();
        let prep = (|| -> Result<(Glm52TpLlBuffer, Glm52TpLlBuffer)> {
            ensure!(rank < ranks, "{topology:?} rank {rank} out of range");
            ensure!(
                slots > 0 && ar_slots > 0,
                "{topology:?} needs at least one layer slot (moe {slots}, ar {ar_slots})"
            );
            ensure!(
                device_ordinal < ranks,
                "{topology:?} device ordinal {device_ordinal} outside its fleet"
            );
            ensure!(
                exchange.rank_count == ranks,
                "{topology:?} exchange has {} ranks, expected {ranks}",
                exchange.rank_count
            );
            let fleet: Vec<usize> = (0..ranks).collect();
            let rs =
                Glm52TpLlBuffer::alloc(topology, slots * topology.rs_slot_packets() * 16, &fleet)?;
            let ar = Glm52TpLlBuffer::alloc(
                topology,
                glm52_tp_ar_buffer_bytes(topology, ar_slots),
                &fleet,
            )?;
            Ok((rs, ar))
        })();
        let vas = prep
            .as_ref()
            .map(|(rs, ar)| Glm52TpLlVas {
                rs: std::array::from_fn(|accessor| {
                    if accessor < ranks {
                        rs.addr_for(accessor)
                    } else {
                        0
                    }
                }),
                ar: std::array::from_fn(|accessor| {
                    if accessor < ranks {
                        ar.addr_for(accessor)
                    } else {
                        0
                    }
                }),
            })
            .map_err(|err| format!("{err:#}"));
        let table = exchange.publish_and_wait(rank, vas)?;
        let (rs, ar) = prep.expect("own failure would have surfaced via publish_and_wait");
        let rs_slot = GLM52_TP_HIDDEN * 16;
        let ar_slot = glm52_tp_ar_chunk_packets(topology) * 16;
        let peer_rs = std::array::from_fn(|peer| {
            if peer < ranks {
                table[peer].rs[device_ordinal] + (rank * rs_slot) as u64
            } else {
                0
            }
        });
        let peer_ar = std::array::from_fn(|peer| {
            if peer < ranks {
                table[peer].ar[device_ordinal] + (rank * ar_slot) as u64
            } else {
                0
            }
        });
        let mut epoch_dev = ctx.stream.alloc_zeros::<u64>(1)?;
        ctx.stream.memcpy_htod(&[1u64], &mut epoch_dev)?;
        let mut active_rows_dev = ctx.stream.alloc_zeros::<i32>(1)?;
        ctx.stream
            .memcpy_htod(&[GLM52_TP_TOKENS as i32], &mut active_rows_dev)?;
        let grid_blocks = glm52_moe_tp_max_blocks(topology)?;
        Ok(Self {
            topology,
            rank,
            ar_slots,
            grid_blocks,
            rs_local: rs.addr_for(device_ordinal),
            ar_local: ar.addr_for(device_ordinal),
            _rs: rs,
            _ar: ar,
            peer_rs,
            peer_ar,
            epoch_dev,
            active_rows_dev,
            guidx: ctx.stream.alloc_zeros(GLM52_TP_UNION_MAX)?,
            guprob: ctx.stream.alloc_zeros(topology.guprob_len())?,
            gucnt: ctx.stream.alloc_zeros(1)?,
            gused: ctx.stream.alloc_zeros(EXPERTS)?,
            ug: ctx.stream.alloc_zeros(topology.ug_len())?,
            cpart: ctx.stream.alloc_zeros(topology.cpart_len())?,
        })
    }

    pub(crate) fn advance_epoch(&mut self, ctx: &DeviceContext) -> Result<()> {
        glm52_moe_tp_epoch_advance(ctx, self.topology, &mut self.epoch_dev)
    }

    pub(crate) fn stage_active_rows(&mut self, ctx: &DeviceContext, active: usize) -> Result<()> {
        ensure!(
            active <= GLM52_TP_TOKENS,
            "{:?} active rows {active} exceeds the bucket {GLM52_TP_TOKENS}",
            self.topology
        );
        ctx.stream
            .memcpy_htod(&[active as i32], &mut self.active_rows_dev)?;
        Ok(())
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn ranks(&self) -> usize {
        self.topology.ranks()
    }

    pub(crate) fn attn_ar_launch(
        &mut self,
        ctx: &DeviceContext,
        layer_slot: usize,
        rows: usize,
        partial: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(
            layer_slot < self.ar_slots,
            "{:?} AR slot {layer_slot} outside allocated {} slots",
            self.topology,
            self.ar_slots
        );
        glm52_tp_ar_launch(
            ctx,
            self.topology,
            layer_slot,
            rows,
            partial,
            out,
            self.ar_local,
            self.peer_ar,
            &self.epoch_dev,
            Some(&self.active_rows_dev),
            self.rank,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        bank: &Glm52MoeTpSliceBank,
        normed2: &CudaSlice<bf16>,
        topk_idx: &CudaSlice<i32>,
        topk_prob: &CudaSlice<f32>,
        mlp_out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        debug_assert_eq!(GLM52_HIDDEN, H);
        ensure!(
            bank.tp_ranks == self.topology.ranks()
                && bank.slice_i == self.topology.slice_i()
                && bank.slice_rows == self.topology.slice_rows(),
            "{:?} launcher received TP{} slice geometry (slice_i={}, slice_rows={})",
            self.topology,
            bank.tp_ranks,
            bank.slice_i,
            bank.slice_rows
        );
        let mut bufs = Glm52MoeTpBuffers {
            guidx: &mut self.guidx,
            guprob: &mut self.guprob,
            gucnt: &mut self.gucnt,
            gused: &mut self.gused,
            ug: &mut self.ug,
            cpart: &mut self.cpart,
            rs_local: self.rs_local,
            peer_rs: self.peer_rs,
            epoch_dev: &mut self.epoch_dev,
            active_rows: Some(&self.active_rows_dev),
        };
        glm52_moe_tp_layer_launch(
            ctx,
            self.topology,
            slot,
            normed2,
            topk_idx,
            topk_prob,
            &bank.w13,
            &bank.w13_scale,
            &bank.w2,
            &bank.w2_scale,
            mlp_out,
            &mut bufs,
            self.rank,
            self.grid_blocks,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_staging_geometry() {
        // A synthetic expert with row-index-stamped bytes must land at the
        // right slice offsets for every rank.
        let mut w13_src = vec![0u8; INTERMEDIATE * H];
        for (row, chunk) in w13_src.chunks_mut(H).enumerate() {
            chunk.fill((row / SLICE_I) as u8); // stamp = owning rank
        }
        let mut w2_src = vec![0u8; W2_N * W2_K];
        for (row, chunk) in w2_src.chunks_mut(W2_K).enumerate() {
            for (col_block, seg) in chunk.chunks_mut(SLICE_I).enumerate() {
                seg.fill((row % 251) as u8 ^ (col_block as u8));
            }
        }
        for rank in 0..RANKS {
            let mut s = SliceStaging::new(rank, RANKS).expect("TP8 slice staging");
            s.put_w13_weight(3, false, &w13_src);
            s.put_w13_weight(3, true, &w13_src);
            s.put_w2_weight(3, &w2_src);
            let base = 3 * SLICE_ROWS * H;
            assert!(
                s.w13[base..base + SLICE_ROWS * H]
                    .iter()
                    .all(|&b| b == rank as u8)
            );
            let w2_base = 3 * H * SLICE_I;
            for row in [0usize, 17, H - 1] {
                let expect = (row % 251) as u8 ^ (rank as u8);
                assert!(
                    s.w2[w2_base + row * SLICE_I..w2_base + (row + 1) * SLICE_I]
                        .iter()
                        .all(|&b| b == expect),
                    "rank {rank} row {row}"
                );
            }
        }
    }

    #[test]
    fn scale_staging_geometry() {
        let scale_f32 = |v: f32| v.to_le_bytes();
        // gate/up scale [16, 48]: value = row block index.
        let mut w13s = vec![0u8; (INTERMEDIATE / QUANT_GROUP) * (H / QUANT_GROUP) * 4];
        for block in 0..INTERMEDIATE / QUANT_GROUP {
            for col in 0..H / QUANT_GROUP {
                let off = (block * (H / QUANT_GROUP) + col) * 4;
                w13s[off..off + 4].copy_from_slice(&scale_f32(block as f32));
            }
        }
        // down scale [48, 16]: value = col block index.
        let mut w2s = vec![0u8; W2_SCALE_ROWS * W2_SCALE_COLS * 4];
        for row in 0..W2_SCALE_ROWS {
            for col in 0..W2_SCALE_COLS {
                let off = (row * W2_SCALE_COLS + col) * 4;
                w2s[off..off + 4].copy_from_slice(&scale_f32(col as f32));
            }
        }
        for rank in 0..RANKS {
            let mut s = SliceStaging::new(rank, RANKS).expect("TP8 slice staging");
            s.put_w13_scale(0, false, &w13s);
            s.put_w13_scale(0, true, &w13s);
            s.put_w2_scale(0, &w2s);
            let read_f32 = |bytes: &[u8], idx: usize| {
                f32::from_le_bytes(bytes[idx * 4..idx * 4 + 4].try_into().unwrap())
            };
            // slice blocks 0..2 = gate blocks 2r..2r+2; 2..4 = up (same rows).
            for b in 0..4 {
                let expect = (2 * rank + b % 2) as f32;
                assert_eq!(read_f32(&s.w13_scale, b * 48), expect, "rank {rank} b {b}");
            }
            for row in [0usize, 47] {
                for b in 0..2 {
                    let expect = (2 * rank + b) as f32;
                    assert_eq!(read_f32(&s.w2_scale, row * 2 + b), expect);
                }
            }
        }
    }

    #[test]
    fn tp4_slice_staging_geometry() {
        let mut w13_src = vec![0u8; INTERMEDIATE * H];
        let tp4_slice_i = INTERMEDIATE / 4;
        for (row, chunk) in w13_src.chunks_mut(H).enumerate() {
            chunk.fill((row / tp4_slice_i) as u8);
        }
        let mut w2_src = vec![0u8; W2_N * W2_K];
        for chunk in w2_src.chunks_mut(W2_K) {
            for (col_block, seg) in chunk.chunks_mut(tp4_slice_i).enumerate() {
                seg.fill(col_block as u8);
            }
        }

        for rank in 0..4 {
            let mut s = SliceStaging::new(rank, 4).expect("TP4 slice staging");
            assert_eq!(s.slice_i, 512);
            assert_eq!(s.slice_rows, 1024);
            s.put_w13_weight(2, false, &w13_src);
            s.put_w13_weight(2, true, &w13_src);
            s.put_w2_weight(2, &w2_src);

            let base = 2 * s.slice_rows * H;
            assert!(
                s.w13[base..base + s.slice_rows * H]
                    .iter()
                    .all(|&b| b == rank as u8)
            );
            let w2_base = 2 * H * s.slice_i;
            for row in [0usize, 17, H - 1] {
                assert!(
                    s.w2[w2_base + row * s.slice_i..w2_base + (row + 1) * s.slice_i]
                        .iter()
                        .all(|&b| b == rank as u8),
                    "rank {rank} row {row}"
                );
            }
        }
    }
}
