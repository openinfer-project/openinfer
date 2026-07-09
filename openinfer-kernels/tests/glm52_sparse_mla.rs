//! GLM5.2 right-sized sparse MLA decode parity gate (M5b).
//!
//! The TileLang main kernel is sm_90a-only, so this gate runs on sm90 only
//! (elsewhere it prints a skip). There the new kernel is checked against a
//! naive f64 attention over the same packed fp8_ds_mla cache and against
//! the FlashMLA sparse decode it replaces. The
//! primary assertion is the mean-normalized delta (max-norm on near-zero
//! bf16 outputs is the wrong metric — M4 twin-gate lesson); the FlashMLA leg
//! also requires the new kernel to sit no further from the f64 ground truth
//! than FlashMLA itself does.
//!
//!   cargo test --release -p openinfer-kernels --features glm52 \
//!     --test glm52_sparse_mla -- --ignored --nocapture

#![cfg(feature = "glm52")]

use anyhow::{Result, ensure};
use half::bf16;
use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_HEADS, GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
    GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM, GLM52_FLASHMLA_SPARSE_V_HEAD_DIM, Glm52FlashMlaSparseDecode,
    Glm52SparseMlaDecode, glm52_flashmla_sparse_decode_launch,
    glm52_flashmla_sparse_decode_metadata_launch, glm52_flashmla_sparse_decode_num_sm_parts,
    glm52_sparse_mla_decode_launch, glm52_sparse_mla_reference_launch,
};
use openinfer_kernels::tensor::DeviceContext;

const HEADS_FULL: usize = GLM52_FLASHMLA_SPARSE_HEADS; // 64 query slots
const DQK: usize = GLM52_FLASHMLA_SPARSE_QK_HEAD_DIM; // 576
const DV: usize = GLM52_FLASHMLA_SPARSE_V_HEAD_DIM; // 512
const CACHE_BYTES: usize = 656;
const SM_SCALE: f32 = 0.0625; // GLM52_SM_SCALE (config.rs)
const MAX_SLOTS: usize = 8192;

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Packed fp8_ds_mla cache filled with random-but-valid tokens: e4m3 bytes
/// (NaN patterns 0x7f/0xff excluded), 4 positive group scales, bf16 rope.
/// Random e4m3 bytes make the dequant exact in every implementation — no
/// host-side float->e4m3 encoder to get subtly wrong.
fn build_cache(rng: &mut XorShift) -> Vec<u8> {
    let mut cache = vec![0u8; MAX_SLOTS * CACHE_BYTES];
    for slot in 0..MAX_SLOTS {
        let token = &mut cache[slot * CACHE_BYTES..(slot + 1) * CACHE_BYTES];
        for byte in token.iter_mut().take(DV) {
            let mut b = (rng.next() >> 32) as u8;
            if b & 0x7f == 0x7f {
                b &= 0xbf; // fold the NaN pattern onto a finite value
            }
            *byte = b;
        }
        for g in 0..4 {
            let scale = 0.002f32 * (0.5 + 1.5 * (rng.unit() * 0.5 + 0.5));
            token[512 + 4 * g..512 + 4 * (g + 1)].copy_from_slice(&scale.to_le_bytes());
        }
        for r in 0..64 {
            let v = bf16::from_f32(rng.unit());
            token[528 + 2 * r..528 + 2 * (r + 1)].copy_from_slice(&v.to_le_bytes());
        }
    }
    cache
}

/// Full-width query with real values in slots 0..heads and zero pads above —
/// exactly what glm52_mla_query_assemble leaves in the buffer.
fn build_q(rng: &mut XorShift, batch: usize, heads: usize) -> Vec<bf16> {
    let mut q = vec![bf16::ZERO; batch * HEADS_FULL * DQK];
    for row in 0..batch {
        for h in 0..heads {
            for d in 0..DQK {
                q[(row * HEADS_FULL + h) * DQK + d] = bf16::from_f32(rng.unit());
            }
        }
    }
    q
}

/// Valid-prefix index rows (the DSA indexer's layout): `valid[row]` random
/// slots then -1 padding.
fn build_indices(rng: &mut XorShift, topk: usize, valid: &[usize]) -> Vec<i32> {
    let mut indices = vec![-1i32; valid.len() * topk];
    for (row, &n) in valid.iter().enumerate() {
        for j in 0..n.min(topk) {
            indices[row * topk + j] = (rng.next() % MAX_SLOTS as u64) as i32;
        }
    }
    indices
}

struct Delta {
    mean_norm: f64,
    max_abs: f64,
}

/// Mean-normalized delta over the real head slots.
fn delta(a: &[bf16], b: &[bf16], batch: usize, heads: usize) -> Delta {
    let mut sum_delta = 0.0f64;
    let mut sum_ref = 0.0f64;
    let mut max_abs = 0.0f64;
    for row in 0..batch {
        for h in 0..heads {
            for d in 0..DV {
                let i = (row * HEADS_FULL + h) * DV + d;
                let da = f64::from(a[i].to_f32());
                let db = f64::from(b[i].to_f32());
                sum_delta += (da - db).abs();
                sum_ref += db.abs();
                max_abs = max_abs.max((da - db).abs());
            }
        }
    }
    Delta {
        mean_norm: sum_delta / sum_ref.max(f64::EPSILON),
        max_abs,
    }
}

struct Rig {
    ctx: DeviceContext,
    cache: cudarc::driver::CudaSlice<u8>,
    is_sm90: bool,
    flash_sm_parts: Option<usize>,
}

impl Rig {
    fn new() -> Result<Self> {
        let ctx = DeviceContext::new()?;
        let mut rng = XorShift(0x5eed_ca11_0000_0001);
        let cache_host = build_cache(&mut rng);
        let mut cache = ctx.stream.alloc_zeros::<u8>(cache_host.len())?;
        ctx.stream.memcpy_htod(&cache_host, &mut cache)?;
        let is_sm90 = ctx.ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )? == 9;
        // The FlashMLA leg only exists on sm90 — and there a failing
        // sm_parts query is a real failure, not an architecture skip.
        let flash_sm_parts = is_sm90
            .then(glm52_flashmla_sparse_decode_num_sm_parts)
            .transpose()?;
        Ok(Self {
            ctx,
            cache,
            is_sm90,
            flash_sm_parts,
        })
    }

    fn run_case(
        &mut self,
        label: &str,
        batch: usize,
        heads: usize,
        topk: usize,
        valid: &[usize],
    ) -> Result<()> {
        assert_eq!(valid.len(), batch);
        let mut rng = XorShift(
            0xd15c_0b01_0000_0000 ^ (batch as u64) << 32 ^ (heads as u64) << 16 ^ topk as u64,
        );
        let q_host = build_q(&mut rng, batch, heads);
        let indices_host = build_indices(&mut rng, topk, valid);

        let ctx = &self.ctx;
        let mut q = ctx.stream.alloc_zeros::<bf16>(q_host.len())?;
        ctx.stream.memcpy_htod(&q_host, &mut q)?;
        let mut indices = ctx.stream.alloc_zeros::<i32>(indices_host.len())?;
        ctx.stream.memcpy_htod(&indices_host, &mut indices)?;

        let contract = Glm52SparseMlaDecode {
            batch_size: batch,
            num_blocks: MAX_SLOTS / GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
            topk,
            heads,
            sm_scale: SM_SCALE,
        };
        let mut o_part = ctx.stream.alloc_zeros::<f32>(contract.o_part_len())?;
        let mut ml_part = ctx.stream.alloc_zeros::<f32>(contract.ml_part_len())?;
        let mut latent_new = ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?;
        let mut latent_ref = ctx.stream.alloc_zeros::<bf16>(contract.latent_len())?;

        glm52_sparse_mla_decode_launch(
            ctx,
            contract,
            &q,
            &self.cache,
            &indices,
            &mut o_part,
            &mut ml_part,
            &mut latent_new,
        )?;
        glm52_sparse_mla_reference_launch(
            ctx,
            contract,
            &q,
            &self.cache,
            &indices,
            &mut latent_ref,
        )?;
        let new_host = ctx.stream.clone_dtoh(&latent_new)?;
        let ref_host = ctx.stream.clone_dtoh(&latent_ref)?;
        ctx.stream.synchronize()?;

        let new_vs_ref = delta(&new_host, &ref_host, batch, heads);
        let all_invalid = valid.iter().all(|&n| n == 0);
        println!(
            "{label:>28}: new-vs-ref mean-norm {:.3e} (max {:.3e})",
            new_vs_ref.mean_norm, new_vs_ref.max_abs
        );

        if all_invalid {
            ensure!(
                new_host.iter().all(|v| v.to_f32() == 0.0),
                "{label}: all-invalid rows must combine to exact zeros"
            );
        } else {
            ensure!(
                new_vs_ref.mean_norm < 2e-2,
                "{label}: new-vs-ref mean-norm {:.3e} over gate 2e-2",
                new_vs_ref.mean_norm
            );
        }

        if let Some(num_sm_parts) = self.flash_sm_parts {
            let flash = Glm52FlashMlaSparseDecode {
                batch_size: batch,
                num_blocks: MAX_SLOTS / GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
                topk,
                num_sm_parts,
                sm_scale: SM_SCALE,
            };
            let mut sched = ctx
                .stream
                .alloc_zeros::<i32>(flash.tile_scheduler_metadata_len())?;
            let mut num_splits = ctx.stream.alloc_zeros::<i32>(flash.num_splits_len())?;
            glm52_flashmla_sparse_decode_metadata_launch(
                ctx,
                batch,
                topk,
                num_sm_parts,
                &mut sched,
                &mut num_splits,
            )?;
            let mut latent_flash = ctx.stream.alloc_zeros::<bf16>(flash.latent_len())?;
            let mut lse = ctx.stream.alloc_zeros::<f32>(flash.lse_len())?;
            let mut lse_accum = ctx.stream.alloc_zeros::<f32>(flash.lse_accum_len())?;
            let mut o_accum = ctx.stream.alloc_zeros::<f32>(flash.o_accum_len())?;
            glm52_flashmla_sparse_decode_launch(
                ctx,
                flash,
                &q,
                &self.cache,
                &indices,
                &sched,
                &num_splits,
                &mut latent_flash,
                &mut lse,
                &mut lse_accum,
                &mut o_accum,
            )?;
            let flash_host = ctx.stream.clone_dtoh(&latent_flash)?;
            ctx.stream.synchronize()?;
            let new_vs_flash = delta(&new_host, &flash_host, batch, heads);
            let flash_vs_ref = delta(&flash_host, &ref_host, batch, heads);
            println!(
                "{label:>28}: new-vs-flash {:.3e}, flash-vs-ref {:.3e}",
                new_vs_flash.mean_norm, flash_vs_ref.mean_norm
            );
            if !all_invalid {
                ensure!(
                    new_vs_flash.mean_norm < 2e-2,
                    "{label}: new-vs-flash mean-norm {:.3e} over gate 2e-2",
                    new_vs_flash.mean_norm
                );
                // The replacement must sit no further from the f64 ground
                // truth than the kernel it replaces (2x headroom for split
                // boundary/rounding differences).
                ensure!(
                    new_vs_ref.mean_norm < flash_vs_ref.mean_norm * 2.0 + 1e-4,
                    "{label}: new-vs-ref {:.3e} much worse than flash-vs-ref {:.3e}",
                    new_vs_ref.mean_norm,
                    flash_vs_ref.mean_norm
                );
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires an sm90 GPU (the TileLang main kernel is sm_90a-only)"]
fn sparse_mla_parity_gate() -> Result<()> {
    let mut rig = Rig::new()?;
    if !rig.is_sm90 {
        println!("(not sm90: TileLang main kernel is a NOT_SUPPORTED stub, skipping)");
        return Ok(());
    }

    // Production shape: bucket-8, TP8 shard (8 real heads), full top-2048.
    rig.run_case("b8 h8 topk2048 full", 8, 8, 2048, &[2048; 8])?;
    // Short-context rows: valid prefix + -1 padding, mixed per row.
    rig.run_case(
        "b8 h8 topk2048 diluted",
        8,
        8,
        2048,
        &[2048, 1024, 256, 64, 1, 512, 128, 1536],
    )?;
    // Want-mask pad rows carry arbitrary staged indices; all -1 must be zeros.
    rig.run_case("b8 h8 all-invalid", 8, 8, 2048, &[0; 8])?;
    // Mixed: one dead row among live ones.
    rig.run_case(
        "b8 h8 one dead row",
        8,
        8,
        2048,
        &[2048, 0, 2048, 2048, 0, 2048, 2048, 2048],
    )?;
    // Head-count edges.
    rig.run_case("b4 h16 topk2048", 4, 16, 2048, &[2048; 4])?;
    rig.run_case("b1 h1 topk2048", 1, 1, 2048, &[2048])?;
    // (The topk-256 short tier was dropped; if it comes back, re-add the
    // topk256 cases — they exercise the bound-masked partial gather stage.)
    Ok(())
}
