//! GLM5.2 bucket-1 TP8 MoE: per-rank 1/8-intermediate slices of ALL 257
//! experts (shared folded at bank index 256) + the whole-layer cooperative
//! kernel state (LL packet buffers, cross-rank pointer exchange, scratch
//! arena). Design: `docs/models/glm52/moe-tp8-low-latency.md`.
//!
//! TP8 slices are a second-pass load: the streaming loader only brings this
//! rank's 32-expert EP8 bundle, so pilot layers re-read every expert's
//! checkpoint tensors and keep BOTH banks resident (bucket-1 graphs take the
//! TP8 kernel, larger buckets keep the EP8 dispatch/combine chain).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Condvar, Mutex};

use anyhow::{Context, Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_TP8_AG_BUF_PACKETS, GLM52_TP8_BANK_EXPERTS, GLM52_TP8_BPART_LEN, GLM52_TP8_CPART_LEN,
    GLM52_TP8_HIDDEN, GLM52_TP8_RANKS, GLM52_TP8_RS_BUF_PACKETS, GLM52_TP8_SLICE_I,
    GLM52_TP8_SLICE_ROWS, GLM52_TP8_TOPK, GLM52_TP8_UG_LEN, GLM52_TP8_UNION_MAX,
    Glm52MoeTp8Buffers, Glm52Tp8LlBuffer, glm52_moe_tp8_enable_peer_access,
    glm52_moe_tp8_layer_launch, glm52_moe_tp8_max_blocks,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::config::{GLM52_EXPERT_INTERMEDIATE as INTERMEDIATE, GLM52_HIDDEN};
use crate::moe_decode::{
    Glm52RouterScratch, QUANT_GROUP, W2_K, W2_N, W2_SCALE_COLS, W2_SCALE_ROWS,
};
use crate::weights::{Glm52WeightManifest, expected_tensor_contract, mmap_file, retype_owned};

const H: usize = GLM52_TP8_HIDDEN;
const RANKS: usize = GLM52_TP8_RANKS;
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

/// Cross-rank rendezvous for LL buffer addresses: every rank publishes its
/// (device ordinal, AG address, RS address) and blocks until all 8 are in.
pub(crate) struct Glm52Tp8Exchange {
    slots: Mutex<[Option<(usize, u64, u64)>; RANKS]>,
    all_in: Condvar,
}

impl Glm52Tp8Exchange {
    pub(crate) fn new() -> Self {
        Self {
            slots: Mutex::new([None; RANKS]),
            all_in: Condvar::new(),
        }
    }

    fn publish_and_wait(
        &self,
        rank: usize,
        ordinal: usize,
        ag: u64,
        rs: u64,
    ) -> Result<[(usize, u64, u64); RANKS]> {
        let mut slots = self.slots.lock().expect("TP8 exchange poisoned");
        ensure!(
            slots[rank].is_none(),
            "TP8 exchange rank {rank} published twice"
        );
        slots[rank] = Some((ordinal, ag, rs));
        self.all_in.notify_all();
        while slots.iter().any(Option::is_none) {
            slots = self.all_in.wait(slots).expect("TP8 exchange poisoned");
        }
        Ok(std::array::from_fn(|r| slots[r].expect("checked above")))
    }
}

/// A rank's complete TP8 pilot: the runtime state plus the per-layer slice
/// banks (keyed by absolute layer index). Lives beside the EP8 state in the
/// rank runtime — both topologies stay resident and the bucket size picks
/// the path at capture time.
pub(crate) struct Glm52MoeTp8Rank {
    pub(crate) state: Glm52MoeTp8State,
    pub(crate) slices: BTreeMap<usize, Glm52MoeTp8SliceBank>,
}

/// Per-rank TP8 runtime state: LL buffers, peer pointer tables, scratch
/// arena, and the co-resident grid size. Everything pointer-stable for graph
/// capture. Collective: all ranks' worker threads must construct
/// concurrently with the same `exchange`.
pub(crate) struct Glm52MoeTp8State {
    rank: usize,
    grid_blocks: usize,
    // LL buffers stay alive as long as peers may write into them.
    _ag: Glm52Tp8LlBuffer,
    _rs: Glm52Tp8LlBuffer,
    ag_local: u64,
    rs_local: u64,
    peer_ag: [u64; RANKS],
    peer_rs: [u64; RANKS],
    epoch_dev: CudaSlice<u64>,
    xg: CudaSlice<bf16>,
    topk_all_idx: CudaSlice<i32>,
    topk_all_prob: CudaSlice<f32>,
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
    ) -> Result<Self> {
        ensure!(rank < RANKS, "TP8 rank {rank} out of range");
        let ag = Glm52Tp8LlBuffer::alloc(GLM52_TP8_AG_BUF_PACKETS * 16)?;
        let rs = Glm52Tp8LlBuffer::alloc(GLM52_TP8_RS_BUF_PACKETS * 16)?;
        let table = exchange.publish_and_wait(rank, device_ordinal, ag.addr(), rs.addr())?;
        for &(peer_ordinal, _, _) in &table {
            if peer_ordinal != device_ordinal {
                glm52_moe_tp8_enable_peer_access(peer_ordinal)?;
            }
        }
        // Peer pointers pre-offset to THIS rank's source slot: rank r's push
        // to peer p lands at p's buffer + r * slot.
        let ag_slot = (GLM52_TP8_AG_BUF_PACKETS / RANKS / 2) * 16; // per-rank, per-parity
        let rs_slot = (GLM52_TP8_RS_BUF_PACKETS / RANKS / 2) * 16;
        let peer_ag = std::array::from_fn(|p| table[p].1 + (rank * ag_slot) as u64);
        let peer_rs = std::array::from_fn(|p| table[p].2 + (rank * rs_slot) as u64);
        // Epoch starts at 1: the LL buffers are zeroed, and a zero tag must
        // never match a live epoch.
        let mut epoch_dev = ctx.stream.alloc_zeros::<u64>(1)?;
        ctx.stream.memcpy_htod(&[1u64], &mut epoch_dev)?;
        let grid_blocks = glm52_moe_tp8_max_blocks()?;
        Ok(Self {
            rank,
            grid_blocks,
            ag_local: ag.addr(),
            rs_local: rs.addr(),
            _ag: ag,
            _rs: rs,
            peer_ag,
            peer_rs,
            epoch_dev,
            xg: ctx.stream.alloc_zeros(RANKS * H)?,
            topk_all_idx: ctx.stream.alloc_zeros(RANKS * GLM52_TP8_TOPK)?,
            topk_all_prob: ctx.stream.alloc_zeros(RANKS * GLM52_TP8_TOPK)?,
            guidx: ctx.stream.alloc_zeros(GLM52_TP8_UNION_MAX)?,
            guprob: ctx.stream.alloc_zeros(GLM52_TP8_UNION_MAX * RANKS)?,
            gucnt: ctx.stream.alloc_zeros(1)?,
            gused: ctx.stream.alloc_zeros(256)?,
            bpart: ctx.stream.alloc_zeros(GLM52_TP8_BPART_LEN)?,
            ug: ctx.stream.alloc_zeros(GLM52_TP8_UG_LEN)?,
            cpart: ctx.stream.alloc_zeros(GLM52_TP8_CPART_LEN)?,
        })
    }

    /// Whole-layer TP8 MoE for this rank's single token (bucket-1 only): the
    /// production router already ran (`router.route` holds the top-8), and
    /// `mlp_out` receives routed + shared like the EP8 arm's closing add.
    pub(crate) fn forward(
        &mut self,
        ctx: &DeviceContext,
        bank: &Glm52MoeTp8SliceBank,
        normed2: &CudaSlice<bf16>,
        router: &Glm52RouterScratch,
        mlp_out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        debug_assert_eq!(GLM52_HIDDEN, H);
        let mut bufs = Glm52MoeTp8Buffers {
            xg: &mut self.xg,
            topk_all_idx: &mut self.topk_all_idx,
            topk_all_prob: &mut self.topk_all_prob,
            guidx: &mut self.guidx,
            guprob: &mut self.guprob,
            gucnt: &mut self.gucnt,
            gused: &mut self.gused,
            bpart: &mut self.bpart,
            ug: &mut self.ug,
            cpart: &mut self.cpart,
            ag_local: self.ag_local,
            rs_local: self.rs_local,
            peer_ag: self.peer_ag,
            peer_rs: self.peer_rs,
            epoch_dev: &mut self.epoch_dev,
        };
        glm52_moe_tp8_layer_launch(
            ctx,
            normed2,
            &router.route.topk_idx,
            &router.route.topk_weight,
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
