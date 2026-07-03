//! Single-layer GLM5.2 MLA decode microbenchmarks (bs=1, synthetic weights).
//!
//! The load-only bring-up composed the oracle-validated ops correctness-first:
//! every intermediate is a fresh `alloc_zeros` and every stage its own launch.
//! Before tuning any kernel, this bench measures what that costs on one layer —
//! the whole forward, each stage in isolation, and the allocation share — so
//! the optimization order comes from numbers, not guesses. Weights are
//! synthetic (constant fp8 bytes, unit scales): fp8 GEMMs do no zero-skipping,
//! so latency matches real checkpoints without needing one on disk.
//!
//! Follows the qwen3 `kernel_bench` convention: a `pub` module the
//! feature-gated bench bin drives; model facts live in `config.rs`.

use std::time::{Duration, Instant};

use anyhow::Result;
use cudarc::driver::{CudaEvent, CudaSlice, sys};
use half::bf16;

use openinfer_kernels::ops::{
    GLM52_FLASHMLA_SPARSE_PAGE_SIZE, GLM52_FLASHMLA_SPARSE_TOPK, Glm52FlashMlaSparseDecode,
    Glm52MoeQuantShape, glm52_flashmla_sparse_decode_launch,
    glm52_flashmla_sparse_decode_metadata_launch, glm52_flashmla_sparse_decode_num_sm_parts,
    glm52_fp8_per_token_group_quant_bf16_launch, glm52_mla_cache_pack_launch,
    glm52_mla_query_assemble_launch,
};
use openinfer_kernels::tensor::DeviceContext;

use crate::fp8::{FP8_BLOCK, Glm52ProjBytes, ProjWeight, fp8_linear};
use openinfer_core::cuda_graph::CudaGraphState;

use crate::mla_decode::{
    Glm52MlaDecodeScratch, Glm52MlaLayerWeights, glm52_mla_decode_forward,
    glm52_mla_decode_forward_into,
};

const HEADS: usize = 64;
const HIDDEN: usize = 6144;
const Q_LORA: usize = 2048;
const Q_HEAD: usize = 256;
const KV_LORA: usize = 512;
const KV_A_OUT: usize = 576;
const V_HEAD: usize = 256;
const KV_B_ROWS_PER_HEAD: usize = 448;
const QUERY_DIM: usize = KV_LORA + 64;
const ROPE_HALF: usize = 32;
const CACHE_BYTES_PER_TOKEN: usize = 656;

/// Constant-fill fp8 projection bytes at `[n, k]` with unit block scales.
/// 0x38 is e4m3 1.0 — finite, no NaN patterns, and latency-equivalent to
/// checkpoint bytes for the blockscale GEMM.
fn synth_proj(n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let weight = vec![0x38u8; n * k];
    let scale_elems = n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK);
    let scale: Vec<u8> = (0..scale_elems)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();
    (weight, scale)
}

fn bf16_ones_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .flat_map(|_| bf16::from_f32(1.0).to_le_bytes())
        .collect()
}

/// One synthetic MLA layer plus every forward input, device-resident.
pub struct Glm52MlaDecodeBench {
    pub ctx: DeviceContext,
    weights: Glm52MlaLayerWeights,
    hidden: CudaSlice<bf16>,
    cos: CudaSlice<bf16>,
    sin: CudaSlice<bf16>,
    cache: CudaSlice<u8>,
    topk: CudaSlice<i32>,
    contract: Glm52FlashMlaSparseDecode,
    position: usize,
    start: CudaEvent,
    end: CudaEvent,
}

impl Glm52MlaDecodeBench {
    /// `context_len` is the attended context; topk indices cover
    /// `min(context_len, 2048)` real slots, -1-padded to the fixed 2048.
    pub fn new(context_len: usize) -> Result<Self> {
        anyhow::ensure!(context_len > 0, "context_len must be positive");
        let ctx = DeviceContext::new()?;

        let (qa_w, qa_s) = synth_proj(Q_LORA, HIDDEN);
        let (qb_w, qb_s) = synth_proj(HEADS * Q_HEAD, Q_LORA);
        let (kva_w, kva_s) = synth_proj(KV_A_OUT, HIDDEN);
        let (kvb_w, kvb_s) = synth_proj(HEADS * KV_B_ROWS_PER_HEAD, KV_LORA);
        let (o_w, o_s) = synth_proj(HIDDEN, HEADS * V_HEAD);
        let ln = bf16_ones_bytes(Q_LORA.max(KV_LORA));
        // A `fn` (not a closure) so the returned bytes borrow ties to the
        // input slices' lifetime — a closure can't express that relation.
        fn proj<'a>(w: &'a [u8], s: &'a [u8], n: usize, k: usize) -> Glm52ProjBytes<'a> {
            Glm52ProjBytes {
                weight: w,
                scale: s,
                n,
                k,
            }
        }
        let weights = Glm52MlaLayerWeights::from_host(
            &ctx,
            &proj(&qa_w, &qa_s, Q_LORA, HIDDEN),
            &ln[..Q_LORA * 2],
            &proj(&qb_w, &qb_s, HEADS * Q_HEAD, Q_LORA),
            &proj(&kva_w, &kva_s, KV_A_OUT, HIDDEN),
            &ln[..KV_LORA * 2],
            &proj(&kvb_w, &kvb_s, HEADS * KV_B_ROWS_PER_HEAD, KV_LORA),
            &proj(&o_w, &o_s, HIDDEN, HEADS * V_HEAD),
        )?;

        let hidden = ctx.stream.clone_htod(&vec![bf16::from_f32(0.01); HIDDEN])?;
        let rope: Vec<bf16> = (0..ROPE_HALF)
            .map(|i| bf16::from_f32(((i as f32) * 0.1).cos()))
            .collect();
        let cos = ctx.stream.clone_htod(&rope)?;
        let sin = ctx.stream.clone_htod(&rope)?;

        let position = context_len - 1;
        let num_blocks = context_len.div_ceil(GLM52_FLASHMLA_SPARSE_PAGE_SIZE);
        let cache = ctx
            .stream
            .alloc_zeros(num_blocks * GLM52_FLASHMLA_SPARSE_PAGE_SIZE * CACHE_BYTES_PER_TOKEN)?;
        let real = context_len.min(GLM52_FLASHMLA_SPARSE_TOPK);
        let topk_host: Vec<i32> = (0..GLM52_FLASHMLA_SPARSE_TOPK)
            .map(|i| if i < real { i as i32 } else { -1 })
            .collect();
        let topk = ctx.stream.clone_htod(&topk_host)?;
        let contract = Glm52FlashMlaSparseDecode {
            batch_size: 1,
            num_blocks,
            topk: GLM52_FLASHMLA_SPARSE_TOPK,
            num_sm_parts: glm52_flashmla_sparse_decode_num_sm_parts()?,
            sm_scale: 1.0 / (QUERY_DIM as f32).sqrt(),
        };
        contract.validate()?;

        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let bench = Self {
            ctx,
            weights,
            hidden,
            cos,
            sin,
            cache,
            topk,
            contract,
            position,
            start,
            end,
        };
        bench.ctx.sync()?;
        Ok(bench)
    }

    fn forward_once(&mut self) -> Result<()> {
        let _o = glm52_mla_decode_forward(
            &self.ctx,
            &self.weights,
            &self.hidden,
            &self.cos,
            &self.sin,
            &mut self.cache,
            self.position,
            &self.topk,
            self.contract,
        )?;
        Ok(())
    }

    /// Whole-layer forward: GPU (event) time and wall time per iteration. The
    /// gap between them is host-side cost — dominated by the per-call
    /// `alloc_zeros` chain, which serializes against the device.
    pub fn measure_forward(&mut self, iters: u64) -> Result<(Duration, Duration)> {
        self.forward_once()?;
        self.ctx.sync()?;
        let wall_start = Instant::now();
        let mut gpu_ms = 0.0f64;
        for _ in 0..iters {
            self.start.record(&self.ctx.stream)?;
            self.forward_once()?;
            self.end.record(&self.ctx.stream)?;
            gpu_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }
        self.ctx.sync()?;
        let wall = wall_start.elapsed();
        Ok((Duration::from_secs_f64(gpu_ms / 1_000.0), wall))
    }

    /// Whole-layer forward through the zero-allocation scratch variant —
    /// the as-is vs scratch delta is the per-layer cudaMalloc bill.
    pub fn measure_forward_scratch(&mut self, iters: u64) -> Result<(Duration, Duration)> {
        let mut scratch = Glm52MlaDecodeScratch::new(&self.ctx, self.contract)?;
        glm52_mla_decode_forward_into(
            &self.ctx,
            &self.weights,
            &self.hidden,
            &self.cos,
            &self.sin,
            &mut self.cache,
            self.position,
            &self.topk,
            self.contract,
            &mut scratch,
        )?;
        self.ctx.sync()?;
        let wall_start = Instant::now();
        let mut gpu_ms = 0.0f64;
        for _ in 0..iters {
            self.start.record(&self.ctx.stream)?;
            glm52_mla_decode_forward_into(
                &self.ctx,
                &self.weights,
                &self.hidden,
                &self.cos,
                &self.sin,
                &mut self.cache,
                self.position,
                &self.topk,
                self.contract,
                &mut scratch,
            )?;
            self.end.record(&self.ctx.stream)?;
            gpu_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }
        self.ctx.sync()?;
        let wall = wall_start.elapsed();
        Ok((Duration::from_secs_f64(gpu_ms / 1_000.0), wall))
    }

    /// The scratch forward captured into a CUDA Graph and replayed — collapses
    /// the ~18 per-token kernel launches into a single `cuGraphLaunch`, so the
    /// host-side launch overhead in the wall−gpu gap disappears. The scratch's
    /// pre-allocated buffers and pre-computed tile schedule make the forward a
    /// pure kernel sequence (no alloc, no sync), which capture requires.
    pub fn measure_forward_graph(&mut self, iters: u64) -> Result<(Duration, Duration)> {
        let mut scratch = Glm52MlaDecodeScratch::new(&self.ctx, self.contract)?;
        let mut graph = CudaGraphState::new();
        // First call captures the graph from the real kernel closure.
        graph.run_or_capture(&self.ctx, || {
            glm52_mla_decode_forward_into(
                &self.ctx,
                &self.weights,
                &self.hidden,
                &self.cos,
                &self.sin,
                &mut self.cache,
                self.position,
                &self.topk,
                self.contract,
                &mut scratch,
            )
        })?;
        self.ctx.sync()?;
        let wall_start = Instant::now();
        let mut gpu_ms = 0.0f64;
        for _ in 0..iters {
            self.start.record(&self.ctx.stream)?;
            // exec is instantiated now, so this replays the graph and never
            // calls the closure.
            graph.run_or_capture(&self.ctx, || Ok(()))?;
            self.end.record(&self.ctx.stream)?;
            gpu_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }
        self.ctx.sync()?;
        Ok((
            Duration::from_secs_f64(gpu_ms / 1_000.0),
            wall_start.elapsed(),
        ))
    }

    /// Bitwise parity between the as-is forward and the scratch forward.
    /// Weights and inputs are deterministic, and the scratch path runs the
    /// exact same op sequence, so any mismatch is a real bug, not noise.
    pub fn verify_scratch_parity(&mut self) -> Result<()> {
        let expected = glm52_mla_decode_forward(
            &self.ctx,
            &self.weights,
            &self.hidden,
            &self.cos,
            &self.sin,
            &mut self.cache,
            self.position,
            &self.topk,
            self.contract,
        )?;
        let expected = self.ctx.stream.clone_dtoh(&expected)?;
        let mut scratch = Glm52MlaDecodeScratch::new(&self.ctx, self.contract)?;
        glm52_mla_decode_forward_into(
            &self.ctx,
            &self.weights,
            &self.hidden,
            &self.cos,
            &self.sin,
            &mut self.cache,
            self.position,
            &self.topk,
            self.contract,
            &mut scratch,
        )?;
        let actual = self.ctx.stream.clone_dtoh(scratch.output())?;
        self.ctx.sync()?;
        let mismatches = expected
            .iter()
            .zip(&actual)
            .filter(|(e, a)| e.to_bits() != a.to_bits())
            .count();
        anyhow::ensure!(
            mismatches == 0,
            "scratch forward diverges from as-is forward: {mismatches}/{} elements differ",
            expected.len()
        );
        Ok(())
    }

    /// One fp8 projection in isolation (its own quant + layout + GEMM chain,
    /// allocations included — exactly what the forward pays per projection).
    pub fn measure_projection(&mut self, which: &str, iters: u64) -> Result<Duration> {
        let (w, input_len): (&ProjWeight, usize) = match which {
            "q_a" => (self.weights.q_a(), HIDDEN),
            "q_b" => (self.weights.q_b(), Q_LORA),
            "kv_a" => (self.weights.kv_a(), HIDDEN),
            "o_proj" => (self.weights.o_proj(), HEADS * V_HEAD),
            other => anyhow::bail!("unknown projection `{other}`"),
        };
        let input = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); input_len])?;
        let _ = fp8_linear(&self.ctx, w, &input)?;
        self.ctx.sync()?;
        let wall = Instant::now();
        for _ in 0..iters {
            let _ = fp8_linear(&self.ctx, w, &input)?;
        }
        self.ctx.sync()?;
        Ok(wall.elapsed())
    }

    /// The three assembly-family micro launches, measured together per
    /// iteration (allocations excluded — buffers are reused here, which is
    /// what a scratch-based forward would pay).
    pub fn measure_assembly_family(&mut self, iters: u64) -> Result<Duration> {
        let ql_nope = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); HEADS * KV_LORA])?;
        let q_full = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); HEADS * Q_HEAD])?;
        let kv_c = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); KV_LORA])?;
        let k_pe = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); 64])?;
        let mut query = self.ctx.stream.alloc_zeros::<bf16>(HEADS * QUERY_DIM)?;
        let mut ckv_fp8 = self.ctx.stream.alloc_zeros::<u8>(KV_LORA)?;
        let mut ckv_scales = self.ctx.stream.alloc_zeros::<f32>(KV_LORA / FP8_BLOCK)?;

        let mut launch = |bench: &mut Self,
                          query: &mut CudaSlice<bf16>,
                          ckv_fp8: &mut CudaSlice<u8>,
                          ckv_scales: &mut CudaSlice<f32>|
         -> Result<()> {
            glm52_mla_query_assemble_launch(
                &bench.ctx, &ql_nope, &q_full, 192, Q_HEAD, &bench.cos, &bench.sin, query,
            )?;
            glm52_fp8_per_token_group_quant_bf16_launch(
                &bench.ctx,
                Glm52MoeQuantShape {
                    rows: 1,
                    width: KV_LORA,
                    group_size: FP8_BLOCK,
                },
                &kv_c,
                ckv_fp8,
                ckv_scales,
            )?;
            glm52_mla_cache_pack_launch(
                &bench.ctx,
                ckv_fp8,
                ckv_scales,
                &k_pe,
                &bench.cos,
                &bench.sin,
                &mut bench.cache,
                bench.position,
            )?;
            Ok(())
        };

        launch(self, &mut query, &mut ckv_fp8, &mut ckv_scales)?;
        self.ctx.sync()?;
        let wall = Instant::now();
        for _ in 0..iters {
            launch(self, &mut query, &mut ckv_fp8, &mut ckv_scales)?;
        }
        self.ctx.sync()?;
        Ok(wall.elapsed())
    }

    /// FlashMLA sparse decode with pre-allocated metadata/output buffers
    /// (the attention core a scratch-based forward would pay).
    pub fn measure_flashmla(&mut self, iters: u64) -> Result<Duration> {
        let query = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); HEADS * QUERY_DIM])?;
        let c = self.contract;
        let mut sched = self
            .ctx
            .stream
            .alloc_zeros::<i32>(c.tile_scheduler_metadata_len())?;
        let mut splits = self.ctx.stream.alloc_zeros::<i32>(c.num_splits_len())?;
        let mut latent = self.ctx.stream.alloc_zeros::<bf16>(c.latent_len())?;
        let mut lse = self.ctx.stream.alloc_zeros::<f32>(c.lse_len())?;
        let mut lse_accum = self.ctx.stream.alloc_zeros::<f32>(c.lse_accum_len())?;
        let mut o_accum = self.ctx.stream.alloc_zeros::<f32>(c.o_accum_len())?;

        glm52_flashmla_sparse_decode_metadata_launch(
            &self.ctx,
            c.batch_size,
            c.num_sm_parts,
            &mut sched,
            &mut splits,
        )?;
        glm52_flashmla_sparse_decode_launch(
            &self.ctx,
            c,
            &query,
            &self.cache,
            &self.topk,
            &sched,
            &splits,
            &mut latent,
            &mut lse,
            &mut lse_accum,
            &mut o_accum,
        )?;
        self.ctx.sync()?;
        let wall = Instant::now();
        for _ in 0..iters {
            glm52_flashmla_sparse_decode_metadata_launch(
                &self.ctx,
                c.batch_size,
                c.num_sm_parts,
                &mut sched,
                &mut splits,
            )?;
            glm52_flashmla_sparse_decode_launch(
                &self.ctx,
                c,
                &query,
                &self.cache,
                &self.topk,
                &sched,
                &splits,
                &mut latent,
                &mut lse,
                &mut lse_accum,
                &mut o_accum,
            )?;
        }
        self.ctx.sync()?;
        Ok(wall.elapsed())
    }

    /// FlashMLA sparse decode (metadata + decode) at an overridden
    /// `num_sm_parts`, everything else from the bench contract. bs=1 sparse
    /// top-k=2048 over-splits at the default (one split per SM ⇒ ~16 KV/split
    /// and a 132-way combine reducing 17.3 MB of `o_accum`); this sweeps the
    /// split count to find where the combine round-trip stops paying for the
    /// extra partial parallelism. Returns `None` if the kernel rejects the
    /// count (`validate`/shape guard) so the caller can skip it.
    pub fn measure_flashmla_at(
        &mut self,
        num_sm_parts: usize,
        iters: u64,
    ) -> Result<Option<Duration>> {
        let mut c = self.contract;
        c.num_sm_parts = num_sm_parts;
        if c.validate().is_err() {
            return Ok(None);
        }
        let query = self
            .ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.01); HEADS * QUERY_DIM])?;
        let mut sched = self
            .ctx
            .stream
            .alloc_zeros::<i32>(c.tile_scheduler_metadata_len())?;
        let mut splits = self.ctx.stream.alloc_zeros::<i32>(c.num_splits_len())?;
        let mut latent = self.ctx.stream.alloc_zeros::<bf16>(c.latent_len())?;
        let mut lse = self.ctx.stream.alloc_zeros::<f32>(c.lse_len())?;
        let mut lse_accum = self.ctx.stream.alloc_zeros::<f32>(c.lse_accum_len())?;
        let mut o_accum = self.ctx.stream.alloc_zeros::<f32>(c.o_accum_len())?;
        let run = |b: &mut Self,
                   sched: &mut CudaSlice<i32>,
                   splits: &mut CudaSlice<i32>,
                   latent: &mut CudaSlice<bf16>,
                   lse: &mut CudaSlice<f32>,
                   lse_accum: &mut CudaSlice<f32>,
                   o_accum: &mut CudaSlice<f32>|
         -> Result<()> {
            glm52_flashmla_sparse_decode_metadata_launch(
                &b.ctx,
                c.batch_size,
                c.num_sm_parts,
                sched,
                splits,
            )?;
            glm52_flashmla_sparse_decode_launch(
                &b.ctx, c, &query, &b.cache, &b.topk, sched, splits, latent, lse, lse_accum,
                o_accum,
            )
        };
        run(
            self,
            &mut sched,
            &mut splits,
            &mut latent,
            &mut lse,
            &mut lse_accum,
            &mut o_accum,
        )?;
        self.ctx.sync()?;
        let wall = Instant::now();
        for _ in 0..iters {
            run(
                self,
                &mut sched,
                &mut splits,
                &mut latent,
                &mut lse,
                &mut lse_accum,
                &mut o_accum,
            )?;
        }
        self.ctx.sync()?;
        Ok(Some(wall.elapsed()))
    }

    /// The default `num_sm_parts` the device query picks (the over-split point).
    pub fn default_num_sm_parts(&self) -> usize {
        self.contract.num_sm_parts
    }

    /// Override the split count for every subsequent measurement (validated).
    /// Lets the driver measure the whole graphed forward at the swept optimum,
    /// not just the isolated flashmla stage.
    pub fn set_num_sm_parts(&mut self, parts: usize) -> Result<()> {
        let mut c = self.contract;
        c.num_sm_parts = parts;
        c.validate()?;
        self.contract = c;
        Ok(())
    }

    /// Confirm a swept `num_sm_parts` is a pure parallelization knob, not a
    /// correctness one: run the sparse decode at `parts` and at the device
    /// default, and report the max abs diff of the `latent` output. The split
    /// count only changes the split-KV reduction tree, so outputs must agree
    /// within fp associativity (not bitwise). A large diff means the count is
    /// unsafe and its sweep timing must not be treated as a usable tuning.
    pub fn flashmla_parts_max_diff(&mut self, parts: usize) -> Result<f32> {
        let latent = |b: &mut Self, num_sm_parts: usize| -> Result<Vec<bf16>> {
            let mut c = b.contract;
            c.num_sm_parts = num_sm_parts;
            c.validate()?;
            let query = b
                .ctx
                .stream
                .clone_htod(&vec![bf16::from_f32(0.01); HEADS * QUERY_DIM])?;
            let mut sched = b
                .ctx
                .stream
                .alloc_zeros::<i32>(c.tile_scheduler_metadata_len())?;
            let mut splits = b.ctx.stream.alloc_zeros::<i32>(c.num_splits_len())?;
            let mut out = b.ctx.stream.alloc_zeros::<bf16>(c.latent_len())?;
            let mut lse = b.ctx.stream.alloc_zeros::<f32>(c.lse_len())?;
            let mut lse_accum = b.ctx.stream.alloc_zeros::<f32>(c.lse_accum_len())?;
            let mut o_accum = b.ctx.stream.alloc_zeros::<f32>(c.o_accum_len())?;
            glm52_flashmla_sparse_decode_metadata_launch(
                &b.ctx,
                c.batch_size,
                c.num_sm_parts,
                &mut sched,
                &mut splits,
            )?;
            glm52_flashmla_sparse_decode_launch(
                &b.ctx,
                c,
                &query,
                &b.cache,
                &b.topk,
                &sched,
                &splits,
                &mut out,
                &mut lse,
                &mut lse_accum,
                &mut o_accum,
            )?;
            Ok(b.ctx.stream.clone_dtoh(&out)?)
        };
        let a = latent(self, self.contract.num_sm_parts)?;
        let b = latent(self, parts)?;
        self.ctx.sync()?;
        Ok(a.iter()
            .zip(&b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0f32, f32::max))
    }
}
