//! Device gate for the GLM5.2 weight-only routed-expert chain
//! (tiles metadata + masked grouped bf16×fp8 mma GEMM + weighted SiLU),
//! at the boundary shim shapes: EP4 (64 local experts × 32 global tokens),
//! EP8 (32 × 64), and EP64 (4 × 512, the local-experts < topk extreme).
//!
//! Single GPU, no DeepEP: a synthetic psum_expert drives the tile kernel and
//! a randomized aligned receive layout drives the GEMM/SiLU, checked against
//! an f32 host reference at bf16-output tolerance. Covers first/last expert,
//! empty experts, a 1-row tile, a full 8-row tile, a 9-row expert (two
//! tiles), alignment-gap rows staying untouched, and the real W13/W2 shapes.

#![cfg(feature = "glm52")]

use half::bf16;
use openinfer_kernels::ops::GLM52_MOE_EP_WO_TILE_ROWS;
use openinfer_kernels::ops::Glm52DeepGemmGroupedFp8Kind;
use openinfer_kernels::ops::glm52_moe_ep_wo_masked_mma_launch;
use openinfer_kernels::ops::glm52_moe_ep_wo_max_tiles;
use openinfer_kernels::ops::glm52_moe_ep_wo_silu_launch;
use openinfer_kernels::ops::glm52_moe_ep_wo_tiles_launch;
use openinfer_kernels::tensor::DeviceContext;

const ALIGN: usize = 64;
const TOPK: usize = 8;

/// e4m3 byte -> f32 (matches `__nv_fp8_e4m3` semantics; NaN encodings are
/// filtered out of the test data).
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((byte >> 3) & 0xf) as i32;
    let man = (byte & 0x7) as f32;
    if exp == 0 {
        sign * (man / 8.0) * (2.0f32).powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * (2.0f32).powi(exp - 7)
    }
}

/// Deterministic byte stream (no rand dependency); e4m3 NaN encodings
/// (0x7f/0xff) are remapped.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn fp8_byte(&mut self) -> u8 {
        let b = (self.next_u32() & 0xff) as u8;
        if b & 0x7f == 0x7f { b & 0xf7 } else { b }
    }
    fn unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

struct Layout {
    starts: Vec<usize>,
    counts: Vec<usize>,
    psum: Vec<i32>,
    aligned_end: usize,
    tiles: Vec<(usize, usize, usize)>, // (row base, expert, rows)
}

fn build_layout(groups: usize, expert_counts: &[(usize, usize)]) -> Layout {
    let mut counts = vec![0usize; groups];
    for &(e, c) in expert_counts {
        counts[e] = c;
    }
    let mut starts = vec![0usize; groups];
    let mut psum = vec![0i32; groups];
    let mut cursor = 0usize;
    let mut tiles = Vec::new();
    for e in 0..groups {
        let start = if e == 0 {
            0
        } else {
            cursor.div_ceil(ALIGN) * ALIGN
        };
        starts[e] = start;
        cursor = start + counts[e];
        psum[e] = cursor as i32;
        let mut r = 0;
        while r < counts[e] {
            let rows = GLM52_MOE_EP_WO_TILE_ROWS.min(counts[e] - r);
            tiles.push((start + r, e, rows));
            r += rows;
        }
    }
    let aligned_end = cursor.div_ceil(ALIGN) * ALIGN;
    Layout {
        starts,
        counts,
        psum,
        aligned_end,
        tiles,
    }
}

/// Per-expert real row counts for the synthetic step (everything else 0):
/// 1-row, full-tile, two-tile, mid, and tail-expert cases at each shape.
#[test]
fn glm52_moe_ep_wo_chain_matches_host_reference_ep4_shape() {
    run_chain_case(64, 32, &[(0, 1), (2, 8), (3, 9), (5, 3), (31, 5), (63, 2)]);
}

/// The EP8 shim shape (32 local experts, 8 ranks x 8 slots global) — the
/// GB300 EP8 configuration's chain runs exactly this geometry.
#[test]
fn glm52_moe_ep_wo_chain_matches_host_reference_ep8_shape() {
    run_chain_case(
        32,
        64,
        &[(0, 2), (1, 8), (4, 17), (13, 1), (22, 9), (31, 6)],
    );
}

/// The EP64 shim shape (4 local experts, 64 ranks x 8 slots global): the
/// few-experts/deep-rows extreme where local experts (4) < topk (8) — the
/// EP16/EP32 geometries interpolate between this and the EP8 case.
#[test]
fn glm52_moe_ep_wo_chain_matches_host_reference_ep64_shape() {
    run_chain_case(4, 512, &[(0, 1), (1, 9), (2, 64), (3, 3)]);
}

fn run_chain_case(groups: usize, global_tokens: usize, expert_counts: &[(usize, usize)]) {
    let Ok(ctx) = DeviceContext::new() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let layout = build_layout(groups, expert_counts);
    let max_tiles = glm52_moe_ep_wo_max_tiles(groups, global_tokens, TOPK);
    assert!(layout.tiles.len() <= max_tiles);
    let expanded = layout.aligned_end;
    // m_capacity mirrors the production bound-rows math loosely; anything
    // >= aligned_end is valid for the trap check.
    let m_capacity = expanded;

    // --- tiles kernel ------------------------------------------------------
    let psum_dev = ctx.stream.clone_htod(&layout.psum).expect("psum H2D");
    let mut tiles_dev = ctx
        .stream
        .alloc_zeros::<i32>(2 * max_tiles)
        .expect("tiles alloc");
    let mut count_dev = ctx.stream.alloc_zeros::<i32>(1).expect("count alloc");
    glm52_moe_ep_wo_tiles_launch(
        &ctx,
        groups,
        m_capacity,
        global_tokens,
        max_tiles,
        &psum_dev,
        &mut tiles_dev,
        &mut count_dev,
    )
    .expect("tiles launch");
    let tiles_host = ctx.stream.clone_dtoh(&tiles_dev).expect("tiles D2H");
    let count_host = ctx.stream.clone_dtoh(&count_dev).expect("count D2H");
    assert_eq!(count_host[0] as usize, layout.tiles.len(), "tile count");
    for (i, &(base, expert, rows)) in layout.tiles.iter().enumerate() {
        assert_eq!(tiles_host[2 * i] as usize, base, "tile {i} base");
        assert_eq!(
            tiles_host[2 * i + 1] as usize,
            expert | (rows << 16),
            "tile {i} meta"
        );
    }

    // --- masked mma GEMM (both operand shapes; W2 exercises the per-row
    // route-weight scaling of the f32 accumulator) --------------------------
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let mut row_weights = vec![0f32; expanded];
    for v in row_weights.iter_mut() {
        *v = (rng.unit_f32() + 1.5) * 0.5; // 0.25..1.25
    }
    let row_weights_dev = ctx.stream.clone_htod(&row_weights).expect("rw H2D");
    for kind in [
        Glm52DeepGemmGroupedFp8Kind::W13,
        Glm52DeepGemmGroupedFp8Kind::W2,
    ] {
        let (n, k) = kind.shape();
        let weighted = kind == Glm52DeepGemmGroupedFp8Kind::W2;
        let active: Vec<usize> = expert_counts.iter().map(|&(e, _)| e).collect();
        // Weight bank: random e4m3 for active experts, zeros elsewhere (the
        // kernel must only touch listed experts, and zero banks keep host
        // generation cheap).
        let mut weight = vec![0u8; groups * n * k];
        let mut weight_scale = vec![0f32; groups * (n / 128) * (k / 128)];
        for &e in &active {
            let w = &mut weight[e * n * k..(e + 1) * n * k];
            for b in w.iter_mut() {
                *b = rng.fp8_byte();
            }
            let s = &mut weight_scale[e * (n / 128) * (k / 128)..(e + 1) * (n / 128) * (k / 128)];
            for v in s.iter_mut() {
                *v = 0.5 + (rng.unit_f32() + 1.0) * 0.75; // 0.5..2.0
            }
        }
        let mut act = vec![bf16::ZERO; expanded * k];
        for v in act.iter_mut() {
            *v = bf16::from_f32(rng.unit_f32());
        }

        let act_dev = ctx.stream.clone_htod(&act).expect("act H2D");
        let weight_dev = ctx.stream.clone_htod(&weight).expect("weight H2D");
        let scale_dev = ctx.stream.clone_htod(&weight_scale).expect("scale H2D");
        // Sentinel output: gap rows must keep it.
        let sentinel = bf16::from_f32(-1234.5);
        let out_init = vec![sentinel; expanded * n];
        let mut out_dev = ctx.stream.clone_htod(&out_init).expect("out H2D");
        glm52_moe_ep_wo_masked_mma_launch(
            &ctx,
            kind,
            groups,
            max_tiles,
            &act_dev,
            &weight_dev,
            &scale_dev,
            &tiles_dev,
            &count_dev,
            weighted.then_some(&row_weights_dev),
            &mut out_dev,
        )
        .expect("mma launch");
        let out_host = ctx.stream.clone_dtoh(&out_dev).expect("out D2H");

        // Host reference in f32 (per-128-block partial then scale, matching
        // the kernel's scale association; the mma slot order inside a block
        // differs, so compare at bf16-rounding tolerance).
        let mut max_rel = 0f32;
        for e in 0..groups {
            for r in 0..layout.counts[e] {
                let row = layout.starts[e] + r;
                // Sample output columns rather than the full n for speed.
                for col in (0..n).step_by(37) {
                    let mut acc = 0f64;
                    for kb in (0..k).step_by(128) {
                        let mut partial = 0f64;
                        for kk in kb..kb + 128 {
                            let w = e4m3_to_f32(weight[e * n * k + col * k + kk]) as f64;
                            let x = f64::from(f32::from(act[row * k + kk]));
                            partial += w * x;
                        }
                        let scale = weight_scale
                            [e * (n / 128) * (k / 128) + (col / 128) * (k / 128) + kb / 128];
                        acc += f64::from(scale) * partial;
                    }
                    let got = f32::from(out_host[row * n + col]);
                    let want = if weighted {
                        (acc * f64::from(row_weights[row])) as f32
                    } else {
                        acc as f32
                    };
                    let rel = (got - want).abs() / want.abs().max(1.0);
                    max_rel = max_rel.max(rel);
                    assert!(
                        rel < 2e-2,
                        "{kind:?} row {row} col {col}: got {got}, want {want}"
                    );
                }
            }
            // Alignment-gap rows keep the sentinel.
            let real_end = layout.starts[e] + layout.counts[e];
            let gap_end = if e + 1 < groups {
                layout.starts[e + 1]
            } else {
                expanded
            };
            for row in real_end..gap_end {
                assert_eq!(
                    out_host[row * n],
                    sentinel,
                    "{kind:?} gap row {row} was written"
                );
            }
        }
        eprintln!("{kind:?} max rel err: {max_rel:.2e}");
    }

    // --- SiLU kernel --------------------------------------------------------
    let inter = 2048usize;
    let mut gate_up = vec![bf16::ZERO; expanded * 2 * inter];
    for v in gate_up.iter_mut() {
        *v = bf16::from_f32(rng.unit_f32() * 2.0);
    }
    let gate_up_dev = ctx.stream.clone_htod(&gate_up).expect("gate_up H2D");
    let sentinel = bf16::from_f32(-77.0);
    let mut silu_dev = ctx
        .stream
        .clone_htod(&vec![sentinel; expanded * inter])
        .expect("silu out H2D");
    glm52_moe_ep_wo_silu_launch(
        &ctx,
        inter,
        max_tiles,
        &gate_up_dev,
        &tiles_dev,
        &count_dev,
        &mut silu_dev,
    )
    .expect("silu launch");
    let silu_host = ctx.stream.clone_dtoh(&silu_dev).expect("silu D2H");
    for e in 0..groups {
        for r in 0..layout.counts[e] {
            let row = layout.starts[e] + r;
            for col in (0..inter).step_by(97) {
                let gate = f32::from(gate_up[row * 2 * inter + col]);
                let up = f32::from(gate_up[row * 2 * inter + inter + col]);
                let want = gate * (1.0 / (1.0 + (-gate).exp())) * up;
                let got = f32::from(silu_host[row * inter + col]);
                assert!(
                    (got - want).abs() <= want.abs().max(0.25) * 1e-2,
                    "silu row {row} col {col}: got {got}, want {want}"
                );
            }
        }
        let real_end = layout.starts[e] + layout.counts[e];
        let gap_end = if e + 1 < groups {
            layout.starts[e + 1]
        } else {
            expanded
        };
        for row in real_end..gap_end {
            assert_eq!(silu_host[row * inter], sentinel, "silu gap row {row}");
        }
    }
}
