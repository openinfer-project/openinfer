//! GLM5.2 TP8 MoE: per-rank 1/8-intermediate slices of ALL 257 experts
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
    GLM52_TP8_AR_CHUNK_PACKETS, GLM52_TP8_BANK_EXPERTS, GLM52_TP8_BPART_LEN, GLM52_TP8_CPART_LEN,
    GLM52_TP8_HIDDEN, GLM52_TP8_RANKS, GLM52_TP8_RS_SLOT_PACKETS, GLM52_TP8_SLICE_I,
    GLM52_TP8_SLICE_ROWS, GLM52_TP8_TOKENS, GLM52_TP8_UG_LEN, GLM52_TP8_UNION_MAX,
    Glm52MoeTp8Buffers, Glm52Tp8LlBuffer, glm52_moe_tp8_epoch_advance, glm52_moe_tp8_layer_launch,
    glm52_moe_tp8_max_blocks, glm52_tp8_ar_buffer_bytes, glm52_tp8_ar_launch,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::config::{GLM52_EXPERT_INTERMEDIATE as INTERMEDIATE, GLM52_HIDDEN};
use crate::moe_decode::{EXPERTS, QUANT_GROUP, W2_K, W2_N, W2_SCALE_COLS, W2_SCALE_ROWS};
use crate::weights::{Glm52WeightManifest, expected_tensor_contract, mmap_file, retype_owned};

const H: usize = GLM52_TP8_HIDDEN;
const RANKS: usize = GLM52_TP8_RANKS;

// The replicated shape assumes the scheduler's largest bucket IS the
// kernel's row count: a bigger bucket would pass every >= buffer ensure
// while the kernel silently computes only 8 rows (stale mlp_out on the
// rest).
const _: () =
    assert!(crate::model::GLM52_MAX_BATCH_PER_RANK == openinfer_kernels::ops::GLM52_TP8_TOKENS);
const BANK: usize = GLM52_TP8_BANK_EXPERTS;
const SLICE_ROWS: usize = GLM52_TP8_SLICE_ROWS;
const SLICE_I: usize = GLM52_TP8_SLICE_I;

const W13_SLICE_BYTES: usize = BANK * SLICE_ROWS * H;
const W13_SLICE_SCALE_F32: usize = BANK * (SLICE_ROWS / QUANT_GROUP) * (H / QUANT_GROUP);
const W2_SLICE_BYTES: usize = BANK * H * SLICE_I;
const W2_SLICE_SCALE_F32: usize = BANK * (H / QUANT_GROUP) * (SLICE_I / QUANT_GROUP);

/// One pilot layer's TP8 slice bank: this rank's 1/8-I rows of all 257
/// experts, in the layout the cooperative kernel consumes.
pub(crate) struct Glm52MoeTp8SliceBank {
    pub(crate) w13: CudaSlice<u8>,        // fp8 [257, 512, 6144]
    pub(crate) w13_scale: CudaSlice<f32>, // f32 [257, 4, 48]
    pub(crate) w2: CudaSlice<u8>,         // fp8 [257, 6144, 256]
    pub(crate) w2_scale: CudaSlice<f32>,  // f32 [257, 48, 2]
}

/// Slice one expert's checkpoint tensors into the rank-r staging bank.
/// `bank_idx` is the destination expert slot (routed id, or 256 for shared).
struct SliceStaging {
    rank: usize,
    w13: Vec<u8>,
    w13_scale: Vec<u8>,
    w2: Vec<u8>,
    w2_scale: Vec<u8>,
}

impl SliceStaging {
    fn new(rank: usize) -> Self {
        Self {
            rank,
            w13: vec![0u8; W13_SLICE_BYTES],
            w13_scale: vec![0u8; W13_SLICE_SCALE_F32 * 4],
            w2: vec![0u8; W2_SLICE_BYTES],
            w2_scale: vec![0u8; W2_SLICE_SCALE_F32 * 4],
        }
    }

    /// gate/up [2048, 6144]: rows r*256..(r+1)*256 land at slice rows 0..256
    /// (gate) / 256..512 (up) — one contiguous copy each.
    fn put_w13_weight(&mut self, bank_idx: usize, is_up: bool, src: &[u8]) {
        debug_assert_eq!(src.len(), INTERMEDIATE * H);
        let rows = SLICE_I; // 256 rows per projection per rank
        let src_off = self.rank * rows * H;
        let dst_off = bank_idx * SLICE_ROWS * H + if is_up { SLICE_I * H } else { 0 };
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
        let blocks = SLICE_I / QUANT_GROUP; // 2
        let src_off = self.rank * blocks * row_bytes;
        let dst_off =
            (bank_idx * (SLICE_ROWS / QUANT_GROUP) + if is_up { blocks } else { 0 }) * row_bytes;
        self.w13_scale[dst_off..dst_off + blocks * row_bytes]
            .copy_from_slice(&src[src_off..src_off + blocks * row_bytes]);
    }

    /// down [6144, 2048]: columns r*256..(r+1)*256 of every row — strided
    /// gather into [6144, 256].
    fn put_w2_weight(&mut self, bank_idx: usize, src: &[u8]) {
        debug_assert_eq!(src.len(), W2_N * W2_K);
        let dst_base = bank_idx * H * SLICE_I;
        let src_col = self.rank * SLICE_I;
        for row in 0..H {
            let dst = dst_base + row * SLICE_I;
            let src_off = row * W2_K + src_col;
            self.w2[dst..dst + SLICE_I].copy_from_slice(&src[src_off..src_off + SLICE_I]);
        }
    }

    /// down scale f32 [48, 16]: column blocks 2r..2r+2 of every row block.
    fn put_w2_scale(&mut self, bank_idx: usize, src: &[u8]) {
        debug_assert_eq!(src.len(), W2_SCALE_ROWS * W2_SCALE_COLS * 4);
        let blocks = SLICE_I / QUANT_GROUP; // 2
        let dst_base = bank_idx * W2_SCALE_ROWS * blocks * 4;
        let src_col = self.rank * blocks * 4;
        for row in 0..W2_SCALE_ROWS {
            let dst = dst_base + row * blocks * 4;
            let src_off = row * W2_SCALE_COLS * 4 + src_col;
            self.w2_scale[dst..dst + blocks * 4]
                .copy_from_slice(&src[src_off..src_off + blocks * 4]);
        }
    }

    fn upload(self, ctx: &DeviceContext) -> Result<Glm52MoeTp8SliceBank> {
        let htod = |host: &[u8]| -> Result<CudaSlice<u8>> {
            // SAFETY: fully written by the memcpy below before use.
            let mut dst = unsafe { ctx.stream.alloc::<u8>(host.len()) }?;
            ctx.stream.memcpy_htod(host, &mut dst)?;
            Ok(dst)
        };
        Ok(Glm52MoeTp8SliceBank {
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
pub(crate) fn load_tp8_slice_layer(
    ctx: &DeviceContext,
    model_path: &Path,
    manifest: &Glm52WeightManifest,
    rank: usize,
    layer: usize,
) -> Result<Glm52MoeTp8SliceBank> {
    ensure!(rank < RANKS, "TP8 rank {rank} out of range");
    // (name, bank_idx, projection kind) for all 257 experts x 6 tensors.
    #[derive(Clone, Copy)]
    enum Kind {
        Gate,
        Up,
        Down,
        GateScale,
        UpScale,
        DownScale,
    }
    let mut wanted: Vec<(String, usize, Kind)> = Vec::with_capacity(BANK * 6);
    let prefix = format!("model.layers.{layer}.mlp");
    let push_expert = |stem: String, bank_idx: usize, wanted: &mut Vec<(String, usize, Kind)>| {
        wanted.push((format!("{stem}.gate_proj.weight"), bank_idx, Kind::Gate));
        wanted.push((format!("{stem}.up_proj.weight"), bank_idx, Kind::Up));
        wanted.push((format!("{stem}.down_proj.weight"), bank_idx, Kind::Down));
        wanted.push((
            format!("{stem}.gate_proj.weight_scale_inv"),
            bank_idx,
            Kind::GateScale,
        ));
        wanted.push((
            format!("{stem}.up_proj.weight_scale_inv"),
            bank_idx,
            Kind::UpScale,
        ));
        wanted.push((
            format!("{stem}.down_proj.weight_scale_inv"),
            bank_idx,
            Kind::DownScale,
        ));
    };
    for expert in 0..BANK - 1 {
        push_expert(format!("{prefix}.experts.{expert}"), expert, &mut wanted);
    }
    push_expert(format!("{prefix}.shared_experts"), BANK - 1, &mut wanted);

    let mut by_shard: BTreeMap<String, Vec<(String, usize, Kind)>> = BTreeMap::new();
    for (name, bank_idx, kind) in wanted {
        let shard = manifest.shard_for(&name)?.to_owned();
        by_shard
            .entry(shard)
            .or_default()
            .push((name, bank_idx, kind));
    }

    let mut staging = SliceStaging::new(rank);
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
                "GLM5.2 TP8 tensor {name} contract mismatch: got {:?} {:?}, expected {:?} {:?}",
                view.dtype(),
                view.shape(),
                contract.dtype,
                contract.shape
            );
            let data = view.data();
            match kind {
                Kind::Gate => staging.put_w13_weight(bank_idx, false, data),
                Kind::Up => staging.put_w13_weight(bank_idx, true, data),
                Kind::Down => staging.put_w2_weight(bank_idx, data),
                Kind::GateScale => staging.put_w13_scale(bank_idx, false, data),
                Kind::UpScale => staging.put_w13_scale(bank_idx, true, data),
                Kind::DownScale => staging.put_w2_scale(bank_idx, data),
            }
            placed += 1;
        }
    }
    ensure!(
        placed == BANK * 6,
        "GLM5.2 TP8 layer {layer} slice load placed {placed} tensors, expected {}",
        BANK * 6
    );
    staging.upload(ctx)
}

/// One rank's published LL buffer mappings: per-accessor VA tables (indexed
/// by fleet device ordinal) for its MoE all-reduce buffer and its attention
/// (o_proj epilogue) all-reduce buffer.
#[derive(Clone, Copy)]
struct Glm52Tp8LlVas {
    rs: [u64; RANKS],
    ar: [u64; RANKS],
}

/// Cross-rank rendezvous for LL buffer mappings: every rank publishes its
/// per-accessor VA tables (or its setup failure, as a poison pill) and blocks
/// until all 8 are in. Also owns the shutdown-side rendezvous so no rank
/// unmaps LL buffers a peer's in-flight kernels could still be reading.
pub(crate) struct Glm52Tp8Exchange {
    slots: Mutex<[Option<Result<Glm52Tp8LlVas, String>>; RANKS]>,
    all_in: Condvar,
    departed: Mutex<usize>,
    all_out: Condvar,
}

impl Glm52Tp8Exchange {
    pub(crate) fn new() -> Self {
        Self {
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
        vas: Result<Glm52Tp8LlVas, String>,
    ) -> Result<[Glm52Tp8LlVas; RANKS]> {
        let mut slots = self.slots.lock().expect("TP8 exchange poisoned");
        ensure!(
            slots[rank].is_none(),
            "TP8 exchange rank {rank} published twice"
        );
        slots[rank] = Some(vas);
        self.all_in.notify_all();
        while slots.iter().any(Option::is_none) {
            let (guard, timeout) = self
                .all_in
                .wait_timeout(slots, Duration::from_secs(120))
                .expect("TP8 exchange poisoned");
            slots = guard;
            if timeout.timed_out() && slots.iter().any(Option::is_none) {
                let missing: Vec<usize> = (0..RANKS).filter(|&r| slots[r].is_none()).collect();
                bail!(
                    "TP8 LL rendezvous timed out after 120s — rank(s) {missing:?} never \
                     published (worker died before reaching the exchange?)"
                );
            }
        }
        let failed: Vec<String> = slots
            .iter()
            .enumerate()
            .filter_map(|(r, s)| match s {
                Some(Err(err)) => Some(format!("rank {r}: {err}")),
                _ => None,
            })
            .collect();
        ensure!(
            failed.is_empty(),
            "TP8 LL setup failed on peer rank(s): {}",
            failed.join("; ")
        );
        Ok(std::array::from_fn(|r| {
            *slots[r]
                .as_ref()
                .expect("checked above")
                .as_ref()
                .expect("failures bailed above")
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
        while *departed < RANKS {
            let (guard, timeout) = self
                .all_out
                .wait_timeout(departed, Duration::from_secs(120))
                .expect("TP8 exchange poisoned");
            departed = guard;
            if timeout.timed_out() && *departed < RANKS {
                log::warn!(
                    "GLM5.2 rank {rank} TP8 teardown rendezvous timed out ({}/{RANKS} arrived) \
                     — unmapping anyway; a peer rank likely died",
                    *departed
                );
                return;
            }
        }
    }
}

/// A rank's complete TP8 MoE runtime: the state plus the per-layer slice
/// banks (keyed by absolute layer index).
pub(crate) struct Glm52MoeTp8Rank {
    pub(crate) state: Glm52MoeTp8State,
    pub(crate) slices: BTreeMap<usize, Glm52MoeTp8SliceBank>,
}

impl Glm52MoeTp8Rank {
    /// This layer's TP8 pieces: runtime state, LL slot index (the layer's
    /// position among this rank's sliced layers), and slice bank.
    pub(crate) fn layer_bank(
        &mut self,
        layer: usize,
    ) -> Option<(&mut Glm52MoeTp8State, usize, &Glm52MoeTp8SliceBank)> {
        let slot = self.slices.range(..layer).count();
        let bank = self.slices.get(&layer)?;
        Some((&mut self.state, slot, bank))
    }
}

/// Per-rank TP8 runtime state: LL buffers, peer pointer tables, scratch
/// arena, and the co-resident grid size. Everything pointer-stable for graph
/// capture. Collective: all ranks' worker threads must construct
/// concurrently with the same `exchange`.
pub(crate) struct Glm52MoeTp8State {
    rank: usize,
    grid_blocks: usize,
    // LL buffers stay alive as long as peers may write into them.
    _rs: Glm52Tp8LlBuffer,
    _ar: Glm52Tp8LlBuffer,
    rs_local: u64,
    ar_local: u64,
    peer_rs: [u64; RANKS],
    peer_ar: [u64; RANKS],
    epoch_dev: CudaSlice<u64>,
    // Want-mask: leading-active row count all TP8 kernels of a step read at
    // replay time. Staged host-side once per step (like the old span-owner
    // staging), identically on every rank — LL push/wait symmetry. Production
    // always stages before use (pre-capture stages 0: push nothing, wait on
    // nothing); the full-bucket initial value serves the oracle gates, which
    // launch these kernels without staging.
    active_rows_dev: CudaSlice<i32>,
    guidx: CudaSlice<i32>,
    guprob: CudaSlice<f32>,
    gucnt: CudaSlice<i32>,
    gused: CudaSlice<i32>,
    bpart: CudaSlice<f32>,
    ug: CudaSlice<bf16>,
    cpart: CudaSlice<f32>,
}

impl Glm52MoeTp8State {
    pub(crate) fn new(
        ctx: &DeviceContext,
        rank: usize,
        device_ordinal: usize,
        exchange: &Glm52Tp8Exchange,
        slots: usize,
        ar_slots: usize,
    ) -> Result<Self> {
        // Everything fallible before the exchange runs inside this closure so
        // its error is PUBLISHED as the poison pill — a rank that bails
        // without publishing would hang the other 7 in the rendezvous.
        let prep = (|| -> Result<(Glm52Tp8LlBuffer, Glm52Tp8LlBuffer)> {
            ensure!(rank < RANKS, "TP8 rank {rank} out of range");
            ensure!(
                slots > 0 && ar_slots > 0,
                "TP8 state needs at least one layer slot (moe {slots}, ar {ar_slots})"
            );
            // GLM5.2's DP fleet is device ordinals 0..8 (enforced at launch),
            // so the per-accessor VA tables index directly by device ordinal.
            ensure!(
                device_ordinal < RANKS,
                "TP8 device ordinal {device_ordinal} outside the 0..8 fleet"
            );
            let fleet: Vec<usize> = (0..RANKS).collect();
            let rs = Glm52Tp8LlBuffer::alloc(slots * GLM52_TP8_RS_SLOT_PACKETS * 16, &fleet)?;
            let ar = Glm52Tp8LlBuffer::alloc(glm52_tp8_ar_buffer_bytes(ar_slots), &fleet)?;
            Ok((rs, ar))
        })();
        let vas = prep
            .as_ref()
            .map(|(rs, ar)| Glm52Tp8LlVas {
                rs: std::array::from_fn(|a| rs.addr_for(a)),
                ar: std::array::from_fn(|a| ar.addr_for(a)),
            })
            .map_err(|err| format!("{err:#}"));
        let table = exchange.publish_and_wait(rank, vas)?;
        let (rs, ar) = prep.expect("own failure would have surfaced via publish_and_wait");
        // Peer pointers: THIS device's VA for peer p's buffer (per-accessor
        // mapping — see `Glm52Tp8LlBuffer`), pre-offset to this rank's
        // source-rank slot. The MoE region is [parity][row][src][hidden] (the
        // kernel adds the parity and row strides itself), so the src stride
        // is one hidden row of packets; the AR region's src stride is one
        // chunk of packets (see `GLM52_TP8_AR_CHUNK_PACKETS`).
        let rs_slot = GLM52_TP8_HIDDEN * 16;
        let ar_slot = GLM52_TP8_AR_CHUNK_PACKETS * 16;
        let peer_rs =
            std::array::from_fn(|p| table[p].rs[device_ordinal] + (rank * rs_slot) as u64);
        let peer_ar =
            std::array::from_fn(|p| table[p].ar[device_ordinal] + (rank * ar_slot) as u64);
        // Epoch starts at 1: the LL buffers are zeroed, and a zero tag must
        // never match a live epoch.
        let mut epoch_dev = ctx.stream.alloc_zeros::<u64>(1)?;
        ctx.stream.memcpy_htod(&[1u64], &mut epoch_dev)?;
        let mut active_rows_dev = ctx.stream.alloc_zeros::<i32>(1)?;
        ctx.stream
            .memcpy_htod(&[GLM52_TP8_TOKENS as i32], &mut active_rows_dev)?;
        let grid_blocks = glm52_moe_tp8_max_blocks()?;
        Ok(Self {
            rank,
            grid_blocks,
            rs_local: rs.addr_for(device_ordinal),
            ar_local: ar.addr_for(device_ordinal),
            _rs: rs,
            _ar: ar,
            peer_rs,
            peer_ar,
            epoch_dev,
            active_rows_dev,
            guidx: ctx.stream.alloc_zeros(GLM52_TP8_UNION_MAX)?,
            guprob: ctx.stream.alloc_zeros(GLM52_TP8_UNION_MAX * RANKS)?,
            gucnt: ctx.stream.alloc_zeros(1)?,
            gused: ctx.stream.alloc_zeros(EXPERTS)?,
            bpart: ctx.stream.alloc_zeros(GLM52_TP8_BPART_LEN)?,
            ug: ctx.stream.alloc_zeros(GLM52_TP8_UG_LEN)?,
            cpart: ctx.stream.alloc_zeros(GLM52_TP8_CPART_LEN)?,
        })
    }

    /// Advance the step epoch — exactly once per decode step, before any
    /// layer's `forward` of that step (captured into the same graph).
    pub(crate) fn advance_epoch(&mut self, ctx: &DeviceContext) -> Result<()> {
        glm52_moe_tp8_epoch_advance(ctx, &mut self.epoch_dev)
    }

    /// Stage the want-mask for the next replayed step: rows `[0, active)` of
    /// the bucket are real, the rest are pads (the plan packs actives as a
    /// prefix). Host-side write OUTSIDE the graph — every rank must stage the
    /// same value before replay (pads skip the LL wire on all ranks alike).
    /// Zero is the graph pre-capture shape (all rows pads): the kernels then
    /// push nothing and wait on nothing, so capture pairs trivially.
    pub(crate) fn stage_active_rows(&mut self, ctx: &DeviceContext, active: usize) -> Result<()> {
        ensure!(
            active <= GLM52_TP8_TOKENS,
            "TP8 active rows {active} exceeds the bucket {GLM52_TP8_TOKENS}"
        );
        ctx.stream
            .memcpy_htod(&[active as i32], &mut self.active_rows_dev)?;
        Ok(())
    }

    /// All-reduce `rows` rows of a head-sharded projection partial (the
    /// attention o_proj epilogue): every rank contributes `partial` and ends
    /// with the bit-identical sum in `out`. `layer_slot` is the decoder layer
    /// index (each layer owns one AR slot region); shares the step epoch with
    /// the MoE chain — [`Self::advance_epoch`] once per step covers both.
    pub(crate) fn attn_ar_launch(
        &mut self,
        ctx: &DeviceContext,
        layer_slot: usize,
        rows: usize,
        partial: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        glm52_tp8_ar_launch(
            ctx,
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

    /// One layer's TP8 MoE over ALL `GLM52_TP8_TOKENS` rows (replicated
    /// activations): `normed2`/`topk_*` carry every global row and must be
    /// bit-identical across ranks; `mlp_out` receives all rows of routed +
    /// shared (like the EP8 arm's closing add), bit-identical across ranks.
    /// `slot` is the layer's LL buffer region (its index among this rank's
    /// sliced layers).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &mut self,
        ctx: &DeviceContext,
        slot: usize,
        bank: &Glm52MoeTp8SliceBank,
        normed2: &CudaSlice<bf16>,
        topk_idx: &CudaSlice<i32>,
        topk_prob: &CudaSlice<f32>,
        mlp_out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        debug_assert_eq!(GLM52_HIDDEN, H);
        debug_assert_eq!(GLM52_TP8_TOKENS, RANKS);
        let mut bufs = Glm52MoeTp8Buffers {
            guidx: &mut self.guidx,
            guprob: &mut self.guprob,
            gucnt: &mut self.gucnt,
            gused: &mut self.gused,
            bpart: &mut self.bpart,
            ug: &mut self.ug,
            cpart: &mut self.cpart,
            rs_local: self.rs_local,
            peer_rs: self.peer_rs,
            epoch_dev: &mut self.epoch_dev,
            active_rows: Some(&self.active_rows_dev),
        };
        glm52_moe_tp8_layer_launch(
            ctx,
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
            let mut s = SliceStaging::new(rank);
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
            let mut s = SliceStaging::new(rank);
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
}
