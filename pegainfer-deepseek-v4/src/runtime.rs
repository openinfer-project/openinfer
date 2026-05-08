use std::ptr;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::nccl::{
    ReduceOp,
    safe::{Comm, group_end, group_start},
};
use half::bf16;
use pegainfer_kernels::ffi;

use crate::{
    config::Config,
    model::{
        AttentionWeights, CompressorWeights, ExpertWeights, FfnWeights, IndexerWeights,
        QuantLinearRef, RankWeightView, TensorRef,
    },
    weights::RankGpuContext,
};

pub struct Bf16HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

pub struct Bf16Cache {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub slots: usize,
}

pub struct CompressorDecodeState {
    pub kv: CudaSlice<f32>,
    pub score: CudaSlice<f32>,
    pub hidden_dim: usize,
    pub slots: usize,
}

pub struct LayerDecodeCache {
    pub kv: Bf16Cache,
    pub compressor: Option<CompressorDecodeState>,
    pub indexer_kv: Option<Bf16Cache>,
    pub indexer_compressor: Option<CompressorDecodeState>,
}

pub struct HcHiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub hc: usize,
}

pub struct HcPreState {
    pub raw_mixes: CudaSlice<f32>,
    pub mixes: CudaSlice<f32>,
    pub rms_scales: CudaSlice<f32>,
    pub pre: CudaSlice<f32>,
    pub post: CudaSlice<f32>,
    pub comb: CudaSlice<f32>,
    pub seq_len: usize,
    pub hc: usize,
}

pub struct F32Logits {
    pub data: CudaSlice<f32>,
    pub vocab_size: usize,
}

pub struct RoutedExperts {
    pub weights: CudaSlice<f32>,
    pub indices: CudaSlice<i32>,
    pub topk: usize,
    pub seq_len: usize,
}

pub struct F32HiddenStates {
    pub data: CudaSlice<f32>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

pub struct AttentionProjections {
    pub qr: Bf16HiddenStates,
    pub q: Bf16HiddenStates,
    pub kv: Bf16HiddenStates,
    pub local_heads: usize,
    pub head_dim: usize,
}

pub struct DeepSeekRopeCache {
    pub cos: CudaSlice<f32>,
    pub sin: CudaSlice<f32>,
    pub max_seq_len: usize,
    pub rotary_dim: usize,
}

impl Bf16HiddenStates {
    pub fn zeros(ctx: &RankGpuContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        ctx.set_current()?;
        let data = ctx.stream.alloc_zeros(hidden_dim * seq_len)?;
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }

    pub fn to_host_f32(&self, ctx: &RankGpuContext) -> Result<Vec<f32>> {
        ctx.set_current()?;
        let host = ctx.stream.clone_dtoh(&self.data)?;
        ctx.sync()?;
        Ok(host.iter().map(|value| value.to_f32()).collect())
    }
}

impl Bf16Cache {
    pub fn zeros(ctx: &RankGpuContext, hidden_dim: usize, slots: usize) -> Result<Self> {
        ctx.set_current()?;
        let data = ctx.stream.alloc_zeros(hidden_dim * slots)?;
        Ok(Self {
            data,
            hidden_dim,
            slots,
        })
    }
}

impl CompressorDecodeState {
    pub fn zeros(
        ctx: &RankGpuContext,
        hidden_dim: usize,
        slots: usize,
        score_fill: f32,
    ) -> Result<Self> {
        ctx.set_current()?;
        let kv = ctx.stream.alloc_zeros(hidden_dim * slots)?;
        let score_host = vec![score_fill; hidden_dim * slots];
        let score = ctx.stream.clone_htod(&score_host)?;
        ctx.sync()?;
        Ok(Self {
            kv,
            score,
            hidden_dim,
            slots,
        })
    }
}

impl LayerDecodeCache {
    pub fn zeros(ctx: &RankGpuContext, config: &Config, layer: usize) -> Result<Self> {
        ctx.set_current()?;
        Self::zeros_with_max_seq(ctx, config, layer, config.max_position_embeddings)
    }

    pub fn zeros_with_max_seq(
        ctx: &RankGpuContext,
        config: &Config,
        layer: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            layer < config.compress_ratios.len(),
            "decode cache layer {layer} out of range"
        );
        ensure!(max_seq_len > 0, "decode cache max_seq_len must be positive");
        let ratio = config.compress_ratios[layer];
        let compressed_slots = if ratio > 0 {
            max_seq_len.div_ceil(ratio)
        } else {
            0
        };
        let kv = Bf16Cache::zeros(
            ctx,
            config.head_dim,
            config.sliding_window + compressed_slots,
        )?;
        let compressor = if ratio == 0 {
            None
        } else if ratio == 4 {
            Some(CompressorDecodeState::zeros(
                ctx,
                2 * config.head_dim,
                2 * ratio,
                f32::NEG_INFINITY,
            )?)
        } else {
            Some(CompressorDecodeState::zeros(
                ctx,
                config.head_dim,
                ratio,
                f32::NEG_INFINITY,
            )?)
        };
        let indexer_kv = if ratio == 4 {
            Some(Bf16Cache::zeros(
                ctx,
                config.index_head_dim,
                max_seq_len.div_ceil(ratio),
            )?)
        } else {
            None
        };
        let indexer_compressor = if ratio == 4 {
            Some(CompressorDecodeState::zeros(
                ctx,
                2 * config.index_head_dim,
                2 * ratio,
                f32::NEG_INFINITY,
            )?)
        } else {
            None
        };
        Ok(Self {
            kv,
            compressor,
            indexer_kv,
            indexer_compressor,
        })
    }
}

pub fn copy_bf16_rows_to_cache(
    ctx: &RankGpuContext,
    src: &Bf16HiddenStates,
    cache: &mut Bf16Cache,
    src_start_row: usize,
    dst_start_row: usize,
    rows: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        src.hidden_dim == cache.hidden_dim,
        "copy rows hidden mismatch: src={}, cache={}",
        src.hidden_dim,
        cache.hidden_dim
    );
    ensure!(
        src_start_row + rows <= src.seq_len,
        "copy rows source out of range: start={}, rows={}, seq_len={}",
        src_start_row,
        rows,
        src.seq_len
    );
    ensure!(
        dst_start_row + rows <= cache.slots,
        "copy rows cache out of range: start={}, rows={}, slots={}",
        dst_start_row,
        rows,
        cache.slots
    );
    {
        let (src_ptr, _src_guard) = src.data.device_ptr(&ctx.stream);
        let (dst_ptr, _dst_guard) = cache.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_copy_rows_cuda(
                src_ptr as *const ffi::Half,
                dst_ptr as *mut ffi::Half,
                cache.hidden_dim as i32,
                rows as i32,
                src_start_row as i32,
                dst_start_row as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn copy_window_prefill_to_ring_cache(
    ctx: &RankGpuContext,
    src: &Bf16HiddenStates,
    cache: &mut Bf16Cache,
    window_size: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        cache.slots >= window_size,
        "window cache needs at least {} slots, got {}",
        window_size,
        cache.slots
    );
    let copy_len = src.seq_len.min(window_size);
    if src.seq_len <= window_size {
        copy_bf16_rows_to_cache(ctx, src, cache, 0, 0, copy_len)?;
    } else {
        let cutoff = src.seq_len % window_size;
        let src_start = src.seq_len - window_size;
        let first = window_size - cutoff;
        copy_bf16_rows_to_cache(ctx, src, cache, src_start, cutoff, first)?;
        if cutoff > 0 {
            copy_bf16_rows_to_cache(ctx, src, cache, src_start + first, 0, cutoff)?;
        }
    }
    Ok(())
}

pub(crate) fn copy_bf16_row_to_hidden(
    ctx: &RankGpuContext,
    src: &Bf16HiddenStates,
    row: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        row < src.seq_len,
        "copy row source out of range: row={}, seq_len={}",
        row,
        src.seq_len
    );
    let mut out = Bf16HiddenStates::zeros(ctx, src.hidden_dim, 1)?;
    {
        let (src_ptr, _src_guard) = src.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_copy_rows_cuda(
                src_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                src.hidden_dim as i32,
                1,
                row as i32,
                0,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

impl HcHiddenStates {
    pub fn zeros(
        ctx: &RankGpuContext,
        hidden_dim: usize,
        seq_len: usize,
        hc: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        let data = ctx.stream.alloc_zeros(hidden_dim * seq_len * hc)?;
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
            hc,
        })
    }

    pub fn to_host_f32(&self, ctx: &RankGpuContext) -> Result<Vec<f32>> {
        ctx.set_current()?;
        let host = ctx.stream.clone_dtoh(&self.data)?;
        ctx.sync()?;
        Ok(host.iter().map(|value| value.to_f32()).collect())
    }
}

impl F32Logits {
    pub fn to_host(&self, ctx: &RankGpuContext) -> Result<Vec<f32>> {
        ctx.set_current()?;
        let host = ctx.stream.clone_dtoh(&self.data)?;
        ctx.sync()?;
        Ok(host)
    }
}

pub fn hc_expand_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    hc: usize,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    ensure!(hc > 0, "HC multiplier must be positive");
    let mut out = HcHiddenStates::zeros(ctx, input.hidden_dim, input.seq_len, hc)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_expand_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                hc as i32,
                input.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
        ctx.sync()?;
    }
    Ok(out)
}

pub fn hc_head_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.hc == config.hc_mult,
        "HC head multiplier mismatch: expected {}, got {}",
        config.hc_mult,
        input.hc
    );
    ensure!(
        input.hidden_dim == config.dim,
        "HC head hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC head fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC head scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC head base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );

    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [input.hc, hc_dim],
        "HC head fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [input.hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [1],
        "HC head scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [1],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [input.hc],
        "HC head base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [input.hc],
        hc_base.tensor.shape
    );

    let mut mixes: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut pre: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                ptr::null_mut(),
                ptr::null_mut(),
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                input.hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_head_pre_cuda(
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                pre_ptr as *mut f32,
                input.seq_len as i32,
                input.hc as i32,
                config.hc_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_output_cuda(
                x_ptr as *const ffi::Half,
                pre_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(out)
}

pub fn rank_local_logits_from_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    head: &TensorRef<'_>,
) -> Result<F32Logits> {
    ctx.set_current()?;
    ensure!(
        head.tensor.dtype == safetensors::Dtype::BF16,
        "head weight {} must be BF16, got {:?}",
        head.name,
        head.tensor.dtype
    );
    ensure!(
        head.tensor.shape.len() == 2,
        "head weight {} must be rank-2, got {:?}",
        head.name,
        head.tensor.shape
    );
    let vocab_size = head.tensor.shape[0];
    let hidden_dim = head.tensor.shape[1];
    ensure!(
        hidden_dim == input.hidden_dim,
        "head input dim mismatch: head expects {}, got {}",
        hidden_dim,
        input.hidden_dim
    );
    ensure!(input.seq_len > 0, "logits input seq_len must be positive");

    let mut out = ctx.stream.alloc_zeros(vocab_size)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (head_ptr, _head_guard) = head.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_last_token_bf16_logits_cuda(
                x_ptr as *const ffi::Half,
                head_ptr as *const ffi::Half,
                out_ptr as *mut f32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                vocab_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(F32Logits {
        data: out,
        vocab_size,
    })
}

pub fn final_logits_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    input: &HcHiddenStates,
) -> Result<F32Logits> {
    ctx.set_current()?;
    let hidden = hc_head_bf16_hidden(
        ctx,
        config,
        input,
        &weights.hc_head_fn()?,
        &weights.hc_head_scale()?,
        &weights.hc_head_base()?,
    )?;
    let normed = rms_norm_bf16_hidden(ctx, &hidden, &weights.norm()?, config.rms_norm_eps)?;
    rank_local_logits_from_hidden(ctx, &normed, &weights.head()?)
}

pub fn all_gather_logits_group(
    ranks: &[(&RankGpuContext, &Comm, &F32Logits)],
) -> Result<Vec<F32Logits>> {
    ensure!(
        !ranks.is_empty(),
        "logits all-gather group must contain at least one rank"
    );
    let local_vocab = ranks[0].2.vocab_size;
    ensure!(local_vocab > 0, "local vocab size must be positive");
    for (_, _, logits) in ranks {
        ensure!(
            logits.vocab_size == local_vocab,
            "logits local vocab mismatch: expected {}, got {}",
            local_vocab,
            logits.vocab_size
        );
    }

    let mut gathered = Vec::with_capacity(ranks.len());
    for (ctx, _, _) in ranks {
        gathered.push(F32Logits {
            data: ctx.stream.alloc_zeros(local_vocab * ranks.len())?,
            vocab_size: local_vocab * ranks.len(),
        });
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, comm, local), full) in ranks.iter().zip(gathered.iter_mut()) {
        if let Err(err) = comm.all_gather(&local.data, &mut full.data) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL logits all-gather failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    Ok(gathered)
}

pub fn final_logits_group_bf16_hidden(
    ranks: &[(&RankGpuContext, &RankWeightView<'_>, &Comm, &HcHiddenStates)],
    config: &Config,
) -> Result<Vec<F32Logits>> {
    ensure!(
        !ranks.is_empty(),
        "final logits group must contain at least one rank"
    );
    let mut local = Vec::with_capacity(ranks.len());
    for (ctx, weights, _, input) in ranks {
        local.push(final_logits_rank_local_bf16_hidden(
            ctx, config, weights, input,
        )?);
    }

    let gather_inputs = ranks
        .iter()
        .zip(local.iter())
        .map(|((ctx, _, comm, _), logits)| (*ctx, *comm, logits))
        .collect::<Vec<_>>();
    all_gather_logits_group(&gather_inputs)
}

pub fn hc_pre_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &HcHiddenStates,
    hc_fn: &TensorRef<'_>,
    hc_scale: &TensorRef<'_>,
    hc_base: &TensorRef<'_>,
) -> Result<(Bf16HiddenStates, HcPreState)> {
    ctx.set_current()?;
    ensure!(
        input.hc == config.hc_mult,
        "HC input multiplier mismatch: expected {}, got {}",
        config.hc_mult,
        input.hc
    );
    ensure!(
        input.hidden_dim == config.dim,
        "HC input hidden dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        hc_fn.tensor.dtype == safetensors::Dtype::F32,
        "HC fn {} must be F32, got {:?}",
        hc_fn.name,
        hc_fn.tensor.dtype
    );
    ensure!(
        hc_scale.tensor.dtype == safetensors::Dtype::F32,
        "HC scale {} must be F32, got {:?}",
        hc_scale.name,
        hc_scale.tensor.dtype
    );
    ensure!(
        hc_base.tensor.dtype == safetensors::Dtype::F32,
        "HC base {} must be F32, got {:?}",
        hc_base.name,
        hc_base.tensor.dtype
    );

    let mix_hc = (2 + input.hc) * input.hc;
    let hc_dim = input.hc * input.hidden_dim;
    ensure!(
        hc_fn.tensor.shape == [mix_hc, hc_dim],
        "HC fn {} shape mismatch: expected {:?}, got {:?}",
        hc_fn.name,
        [mix_hc, hc_dim],
        hc_fn.tensor.shape
    );
    ensure!(
        hc_scale.tensor.shape == [3],
        "HC scale {} shape mismatch: expected {:?}, got {:?}",
        hc_scale.name,
        [3],
        hc_scale.tensor.shape
    );
    ensure!(
        hc_base.tensor.shape == [mix_hc],
        "HC base {} shape mismatch: expected {:?}, got {:?}",
        hc_base.name,
        [mix_hc],
        hc_base.tensor.shape
    );

    let mut mixes: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * mix_hc)?;
    let mut raw_mixes: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * mix_hc)?;
    let mut rms_scales: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len)?;
    let mut pre: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut post: CudaSlice<f32> = ctx.stream.alloc_zeros(input.seq_len * input.hc)?;
    let mut comb: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(input.seq_len * input.hc * input.hc)?;
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (fn_ptr, _fn_guard) = hc_fn.tensor.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr_mut(&ctx.stream);
        let (raw_mixes_ptr, _raw_mixes_guard) = raw_mixes.device_ptr_mut(&ctx.stream);
        let (rms_scales_ptr, _rms_scales_guard) = rms_scales.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_mixes_cuda(
                x_ptr as *const ffi::Half,
                fn_ptr as *const f32,
                mixes_ptr as *mut f32,
                raw_mixes_ptr as *mut f32,
                rms_scales_ptr as *mut f32,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                mix_hc as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (mixes_ptr, _mixes_guard) = mixes.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc_scale.tensor.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc_base.tensor.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
        let (post_ptr, _post_guard) = post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _comb_guard) = comb.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_split_sinkhorn_cuda(
                mixes_ptr as *const f32,
                scale_ptr as *const f32,
                base_ptr as *const f32,
                pre_ptr as *mut f32,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                input.seq_len as i32,
                input.hc as i32,
                config.hc_sinkhorn_iters as i32,
                config.hc_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_pre_output_cuda(
                x_ptr as *const ffi::Half,
                pre_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hc as i32,
                input.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok((
        out,
        HcPreState {
            raw_mixes,
            mixes,
            rms_scales,
            pre,
            post,
            comb,
            seq_len: input.seq_len,
            hc: input.hc,
        },
    ))
}

pub fn hc_post_bf16_hidden(
    ctx: &RankGpuContext,
    branch_out: &Bf16HiddenStates,
    residual: &HcHiddenStates,
    pre_state: &HcPreState,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    ensure!(
        branch_out.hidden_dim == residual.hidden_dim,
        "HC post hidden dim mismatch: branch={}, residual={}",
        branch_out.hidden_dim,
        residual.hidden_dim
    );
    ensure!(
        branch_out.seq_len == residual.seq_len,
        "HC post seq len mismatch: branch={}, residual={}",
        branch_out.seq_len,
        residual.seq_len
    );
    ensure!(
        pre_state.seq_len == branch_out.seq_len,
        "HC post pre-state seq len mismatch: state={}, branch={}",
        pre_state.seq_len,
        branch_out.seq_len
    );
    ensure!(
        pre_state.hc == residual.hc,
        "HC post pre-state multiplier mismatch: state={}, residual={}",
        pre_state.hc,
        residual.hc
    );

    let mut out =
        HcHiddenStates::zeros(ctx, branch_out.hidden_dim, branch_out.seq_len, residual.hc)?;
    {
        let (x_ptr, _x_guard) = branch_out.data.device_ptr(&ctx.stream);
        let (residual_ptr, _residual_guard) = residual.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = pre_state.post.device_ptr(&ctx.stream);
        let (comb_ptr, _comb_guard) = pre_state.comb.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hc_post_cuda(
                x_ptr as *const ffi::Half,
                residual_ptr as *const ffi::Half,
                post_ptr as *const f32,
                comb_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                branch_out.seq_len as i32,
                residual.hc as i32,
                branch_out.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn embedding_rank_local(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    token_ids: &CudaSlice<u32>,
    seq_len: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let embed = weights.embed()?;
    ensure!(
        embed.tensor.shape == [config.vocab_size / 8, config.dim],
        "unexpected embed shape {:?}",
        embed.tensor.shape
    );
    let mut out = Bf16HiddenStates::zeros(ctx, config.dim, seq_len)?;

    {
        let (embed_ptr, _embed_guard) = embed.tensor.data.device_ptr(&ctx.stream);
        let (token_ptr, _token_guard) = token_ids.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let vocab_start = (weights.rank() * (config.vocab_size / 8)) as u32;
        let part_vocab_size = (config.vocab_size / 8) as u32;

        let result = unsafe {
            ffi::embedding_batched_vocab_shard_cuda(
                embed_ptr as *const ffi::Half,
                token_ptr as *const u32,
                out_ptr as *mut ffi::Half,
                config.dim as i32,
                seq_len as i32,
                vocab_start,
                part_vocab_size,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(out)
}

pub fn embedding_vocab_parallel_group(
    ranks: &[(&RankGpuContext, &RankWeightView<'_>, &Comm, &CudaSlice<u32>)],
    config: &Config,
    seq_len: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "embedding group must contain at least one rank"
    );

    let mut hidden = Vec::with_capacity(ranks.len());
    for (ctx, weights, _comm, token_ids) in ranks {
        hidden.push(embedding_rank_local(
            ctx, config, weights, token_ids, seq_len,
        )?);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _), hidden) in ranks.iter().zip(hidden.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL embedding all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    Ok(hidden)
}

pub fn rms_norm_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    weight: &crate::model::TensorRef<'_>,
    eps: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        weight.tensor.dtype == safetensors::Dtype::BF16,
        "RMSNorm weight {} must be BF16, got {:?}",
        weight.name,
        weight.tensor.dtype
    );
    ensure!(
        weight.tensor.shape == [input.hidden_dim],
        "RMSNorm weight {} shape mismatch: expected {:?}, got {:?}",
        weight.name,
        [input.hidden_dim],
        weight.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = weight.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.hidden_dim as i32,
                input.seq_len as i32,
                eps,
                ctx.stream.cu_stream(),
            );
        }
    }
    Ok(out)
}

pub fn fp8_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    linear: &QuantLinearRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        linear.weight.tensor.dtype == safetensors::Dtype::F8_E4M3,
        "FP8 linear weight {} must be F8_E4M3, got {:?}",
        linear.weight.name,
        linear.weight.tensor.dtype
    );
    ensure!(
        linear.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "FP8 linear scale {} must be F8_E8M0, got {:?}",
        linear.scale.name,
        linear.scale.tensor.dtype
    );
    ensure!(
        linear.weight.tensor.shape.len() == 2,
        "FP8 linear weight {} must be rank-2, got {:?}",
        linear.weight.name,
        linear.weight.tensor.shape
    );
    let out_dim = linear.weight.tensor.shape[0];
    let in_dim = linear.weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "FP8 linear input dim mismatch: weight {} expects {}, got {}",
        linear.weight.name,
        in_dim,
        input.hidden_dim
    );
    ensure!(
        linear.scale.tensor.shape == [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        "FP8 linear scale {} shape mismatch: expected {:?}, got {:?}",
        linear.scale.name,
        [out_dim.div_ceil(128), in_dim.div_ceil(128)],
        linear.scale.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, out_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = linear.weight.tensor.data.device_ptr(&ctx.stream);
        let (s_ptr, _s_guard) = linear.scale.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp8_linear_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const u8,
                s_ptr as *const u8,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn fp4_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    linear: &QuantLinearRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        linear.weight.tensor.dtype == safetensors::Dtype::F4,
        "FP4 linear weight {} must be F4, got {:?}",
        linear.weight.name,
        linear.weight.tensor.dtype
    );
    ensure!(
        linear.scale.tensor.dtype == safetensors::Dtype::F8_E8M0,
        "FP4 linear scale {} must be F8_E8M0, got {:?}",
        linear.scale.name,
        linear.scale.tensor.dtype
    );
    ensure!(
        linear.weight.tensor.shape.len() == 2,
        "FP4 linear weight {} must be rank-2, got {:?}",
        linear.weight.name,
        linear.weight.tensor.shape
    );
    let out_dim = linear.weight.tensor.shape[0];
    let in_dim = linear.weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "FP4 linear input dim mismatch: weight {} expects {}, got {}",
        linear.weight.name,
        in_dim,
        input.hidden_dim
    );
    ensure!(
        in_dim.is_multiple_of(32),
        "FP4 linear input dim must be divisible by 32, got {in_dim}"
    );
    ensure!(
        linear.scale.tensor.shape == [out_dim, in_dim / 32],
        "FP4 linear scale {} shape mismatch: expected {:?}, got {:?}",
        linear.scale.name,
        [out_dim, in_dim / 32],
        linear.scale.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, out_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = linear.weight.tensor.data.device_ptr(&ctx.stream);
        let (s_ptr, _s_guard) = linear.scale.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp4_linear_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const u8,
                s_ptr as *const u8,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn bf16_linear_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    weight: &TensorRef<'_>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        weight.tensor.dtype == safetensors::Dtype::BF16,
        "BF16 linear weight {} must be BF16, got {:?}",
        weight.name,
        weight.tensor.dtype
    );
    ensure!(
        weight.tensor.shape.len() == 2,
        "BF16 linear weight {} must be rank-2, got {:?}",
        weight.name,
        weight.tensor.shape
    );
    let out_dim = weight.tensor.shape[0];
    let in_dim = weight.tensor.shape[1];
    ensure!(
        in_dim == input.hidden_dim,
        "BF16 linear input dim mismatch: weight {} expects {}, got {}",
        weight.name,
        in_dim,
        input.hidden_dim
    );

    let mut out = Bf16HiddenStates::zeros(ctx, out_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (w_ptr, _w_guard) = weight.tensor.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_linear_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
        ctx.sync()?;
    }
    Ok(out)
}

pub fn swiglu_clamp_bf16_hidden(
    ctx: &RankGpuContext,
    gate: &Bf16HiddenStates,
    up: &Bf16HiddenStates,
    limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        gate.hidden_dim == up.hidden_dim,
        "SwiGLU hidden dim mismatch: gate={}, up={}",
        gate.hidden_dim,
        up.hidden_dim
    );
    ensure!(
        gate.seq_len == up.seq_len,
        "SwiGLU seq len mismatch: gate={}, up={}",
        gate.seq_len,
        up.seq_len
    );

    let mut out = Bf16HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    {
        let (gate_ptr, _gate_guard) = gate.data.device_ptr(&ctx.stream);
        let (up_ptr, _up_guard) = up.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_swiglu_clamp_cuda(
                gate_ptr as *const ffi::Half,
                up_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (gate.hidden_dim * gate.seq_len) as i32,
                limit,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub(crate) fn swiglu_clamp_weighted_bf16_hidden(
    ctx: &RankGpuContext,
    gate: &Bf16HiddenStates,
    up: &Bf16HiddenStates,
    routed: &RoutedExperts,
    global_expert: usize,
    limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        gate.hidden_dim == up.hidden_dim,
        "weighted SwiGLU hidden dim mismatch: gate={}, up={}",
        gate.hidden_dim,
        up.hidden_dim
    );
    ensure!(
        gate.seq_len == up.seq_len,
        "weighted SwiGLU seq len mismatch: gate={}, up={}",
        gate.seq_len,
        up.seq_len
    );
    ensure!(
        routed.seq_len == gate.seq_len,
        "weighted SwiGLU route seq len mismatch: route={}, gate={}",
        routed.seq_len,
        gate.seq_len
    );

    let mut out = Bf16HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    {
        let (gate_ptr, _gate_guard) = gate.data.device_ptr(&ctx.stream);
        let (up_ptr, _up_guard) = up.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = routed.weights.device_ptr(&ctx.stream);
        let (indices_ptr, _indices_guard) = routed.indices.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_swiglu_clamp_weighted_cuda(
                gate_ptr as *const ffi::Half,
                up_ptr as *const ffi::Half,
                weights_ptr as *const f32,
                indices_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                (gate.hidden_dim * gate.seq_len) as i32,
                gate.hidden_dim as i32,
                routed.topk as i32,
                global_expert as i32,
                limit,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn local_expert_forward_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    expert: &ExpertWeights<'_>,
    swiglu_limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let gate = fp4_linear_bf16_hidden(ctx, input, &expert.w1)?;
    let up = fp4_linear_bf16_hidden(ctx, input, &expert.w3)?;
    let activated = swiglu_clamp_bf16_hidden(ctx, &gate, &up, swiglu_limit)?;
    fp4_linear_bf16_hidden(ctx, &activated, &expert.w2)
}

pub fn local_expert_forward_weighted_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    expert: &ExpertWeights<'_>,
    routed: &RoutedExperts,
    global_expert: usize,
    swiglu_limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let gate = fp4_linear_bf16_hidden(ctx, input, &expert.w1)?;
    let up = fp4_linear_bf16_hidden(ctx, input, &expert.w3)?;
    let activated =
        swiglu_clamp_weighted_bf16_hidden(ctx, &gate, &up, routed, global_expert, swiglu_limit)?;
    fp4_linear_bf16_hidden(ctx, &activated, &expert.w2)
}

pub fn shared_expert_forward_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    swiglu_limit: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let gate = fp8_linear_bf16_hidden(ctx, input, &ffn.shared_w1)?;
    let up = fp8_linear_bf16_hidden(ctx, input, &ffn.shared_w3)?;
    let activated = swiglu_clamp_bf16_hidden(ctx, &gate, &up, swiglu_limit)?;
    fp8_linear_bf16_hidden(ctx, &activated, &ffn.shared_w2)
}

pub fn add_bf16_hidden(
    ctx: &RankGpuContext,
    a: &Bf16HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "add hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len,
        "add seq len mismatch: a={}, b={}",
        a.seq_len,
        b.seq_len
    );

    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (out.hidden_dim * out.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn head_rms_norm_bf16_hidden(
    ctx: &RankGpuContext,
    input: &Bf16HiddenStates,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == num_heads * head_dim,
        "head RMSNorm input dim mismatch: expected {}, got {}",
        num_heads * head_dim,
        input.hidden_dim
    );
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_head_rms_norm_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn attention_project_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
) -> Result<AttentionProjections> {
    ctx.set_current()?;
    let qr = fp8_linear_bf16_hidden(ctx, input, &attn.wq_a)?;
    let qr_norm = rms_norm_bf16_hidden(ctx, &qr, &attn.q_norm, config.rms_norm_eps)?;
    let q_raw = fp8_linear_bf16_hidden(ctx, &qr_norm, &attn.wq_b)?;
    let local_heads = q_raw.hidden_dim / config.head_dim;
    ensure!(
        local_heads * config.head_dim == q_raw.hidden_dim,
        "wq_b output dim {} is not divisible by head_dim {}",
        q_raw.hidden_dim,
        config.head_dim
    );
    let q = head_rms_norm_bf16_hidden(
        ctx,
        &q_raw,
        local_heads,
        config.head_dim,
        config.rms_norm_eps,
    )?;
    let kv_raw = fp8_linear_bf16_hidden(ctx, input, &attn.wkv)?;
    let kv = rms_norm_bf16_hidden(ctx, &kv_raw, &attn.kv_norm, config.rms_norm_eps)?;
    Ok(AttentionProjections {
        qr: qr_norm,
        q,
        kv,
        local_heads,
        head_dim: config.head_dim,
    })
}

pub fn precompute_rope_cache(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    max_seq_len: usize,
) -> Result<DeepSeekRopeCache> {
    ctx.set_current()?;
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    ensure!(max_seq_len > 0, "max_seq_len must be positive");
    let rotary_dim = config.qk_rope_head_dim;
    ensure!(
        rotary_dim.is_multiple_of(2),
        "rotary_dim must be even, got {rotary_dim}"
    );

    let compress = config.compress_ratios[layer] > 0;
    let base = if compress {
        config.compress_rope_theta
    } else {
        config.rope_theta
    };
    let original_seq_len = if compress {
        config.rope_scaling.original_seq_len
    } else {
        0
    };
    let factor = config.rope_scaling.factor;
    let beta_fast = config.rope_scaling.beta_fast as f32;
    let beta_slow = config.rope_scaling.beta_slow as f32;

    let mut inv_freq = Vec::with_capacity(rotary_dim / 2);
    for i in 0..rotary_dim / 2 {
        let exponent = (2 * i) as f32 / rotary_dim as f32;
        inv_freq.push(1.0 / base.powf(exponent));
    }
    if original_seq_len > 0 {
        let find_correction_dim = |num_rotations: f32| -> f32 {
            rotary_dim as f32
                * ((original_seq_len as f32) / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = find_correction_dim(beta_fast).floor().max(0.0);
        let high = find_correction_dim(beta_slow)
            .ceil()
            .min((rotary_dim - 1) as f32);
        let high = if (high - low).abs() < f32::EPSILON {
            high + 0.001
        } else {
            high
        };
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }

    let pairs = rotary_dim / 2;
    let inv_freq_gpu = ctx.stream.clone_htod(&inv_freq)?;
    let mut cos = ctx.stream.alloc_zeros::<f32>(max_seq_len * pairs)?;
    let mut sin = ctx.stream.alloc_zeros::<f32>(max_seq_len * pairs)?;
    {
        let (inv_ptr, _inv_guard) = inv_freq_gpu.device_ptr(&ctx.stream);
        let (cos_ptr, _cos_guard) = cos.device_ptr_mut(&ctx.stream);
        let (sin_ptr, _sin_guard) = sin.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fill_rope_cache_cuda(
                inv_ptr as *const f32,
                cos_ptr as *mut f32,
                sin_ptr as *mut f32,
                max_seq_len as i32,
                pairs as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    ctx.sync()?;
    Ok(DeepSeekRopeCache {
        cos,
        sin,
        max_seq_len,
        rotary_dim,
    })
}

pub fn apply_rope_attention_projections(
    ctx: &RankGpuContext,
    projections: &mut AttentionProjections,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        start_pos + projections.q.seq_len <= rope.max_seq_len,
        "RoPE range [{}..{}) exceeds cache len {}",
        start_pos,
        start_pos + projections.q.seq_len,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= projections.head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        projections.head_dim
    );
    ensure!(
        projections.kv.hidden_dim == projections.head_dim,
        "kv dim {} must equal head_dim {}",
        projections.kv.hidden_dim,
        projections.head_dim
    );

    apply_rope_hidden_in_place(
        ctx,
        &mut projections.q,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
        false,
    )?;
    apply_rope_hidden_in_place(
        ctx,
        &mut projections.kv,
        rope,
        1,
        projections.head_dim,
        start_pos,
        false,
    )?;
    fp8_act_quant_nope_bf16_hidden_in_place(
        ctx,
        &mut projections.kv,
        1,
        projections.head_dim,
        rope.rotary_dim,
        64,
    )?;
    Ok(())
}

pub fn fp8_act_quant_nope_bf16_hidden_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    local_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    block_size: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "FP8 in-place quant hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(
        rotary_dim < head_dim,
        "FP8 in-place quant rotary_dim {} must be smaller than head_dim {}",
        rotary_dim,
        head_dim
    );
    let nope_dim = head_dim - rotary_dim;
    ensure!(
        nope_dim.is_multiple_of(block_size),
        "FP8 in-place quant nope_dim {} must be divisible by block_size {}",
        nope_dim,
        block_size
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_fp8_act_quant_nope_bf16_cuda(
                x_ptr as *mut ffi::Half,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rotary_dim as i32,
                block_size as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn apply_rope_hidden_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
    inverse: bool,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "RoPE hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(
        start_pos + hidden.seq_len <= rope.max_seq_len,
        "RoPE range [{}..{}) exceeds cache len {}",
        start_pos,
        start_pos + hidden.seq_len,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_hidden_cuda(
                x_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                start_pos as i32,
                if inverse { 1 } else { 0 },
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn apply_rope_hidden_strided_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
    position_stride: usize,
    inverse: bool,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        hidden.hidden_dim == local_heads * head_dim,
        "strided RoPE hidden dim mismatch: expected {}, got {}",
        local_heads * head_dim,
        hidden.hidden_dim
    );
    ensure!(position_stride > 0, "position_stride must be positive");
    ensure!(
        start_pos + (hidden.seq_len - 1) * position_stride < rope.max_seq_len,
        "strided RoPE range start={} len={} stride={} exceeds cache len {}",
        start_pos,
        hidden.seq_len,
        position_stride,
        rope.max_seq_len
    );
    ensure!(
        rope.rotary_dim <= head_dim,
        "rotary_dim {} exceeds head_dim {}",
        rope.rotary_dim,
        head_dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let (cos_ptr, _cos_guard) = rope.cos.device_ptr(&ctx.stream);
        let (sin_ptr, _sin_guard) = rope.sin.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_apply_rope_hidden_strided_cuda(
                x_ptr as *mut ffi::Half,
                cos_ptr as *const f32,
                sin_ptr as *const f32,
                hidden.seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope.rotary_dim as i32,
                start_pos as i32,
                position_stride as i32,
                if inverse { 1 } else { 0 },
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn sparse_attention_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    attn: &AttentionWeights<'_>,
) -> Result<Bf16HiddenStates> {
    let (topk_idxs, topk) = window_topk_indices(ctx, projections.q.seq_len, config.sliding_window)?;
    indexed_attention_prefill_bf16_hidden(ctx, config, projections, attn, &topk_idxs, topk)
}

pub fn window_topk_indices(
    ctx: &RankGpuContext,
    seq_len: usize,
    window_size: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "seq_len must be positive");
    ensure!(window_size > 0, "window_size must be positive");
    let topk = seq_len.min(window_size);
    let mut host = vec![-1_i32; seq_len * topk];
    for token in 0..seq_len {
        let key_start = token.saturating_sub(window_size - 1);
        let mut route = 0;
        for key in key_start..=token {
            if route < topk {
                host[token * topk + route] = key as i32;
                route += 1;
            }
        }
    }
    let data = ctx.stream.clone_htod(&host)?;
    ctx.sync()?;
    Ok((data, topk))
}

pub fn window_topk_indices_decode(
    ctx: &RankGpuContext,
    start_pos: usize,
    window_size: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(window_size > 0, "window_size must be positive");
    let mut host = vec![-1_i32; window_size];
    if start_pos >= window_size - 1 {
        let pos = start_pos % window_size;
        let mut route = 0;
        for key in (pos + 1)..window_size {
            host[route] = key as i32;
            route += 1;
        }
        for key in 0..=pos {
            host[route] = key as i32;
            route += 1;
        }
    } else {
        for (idx, slot) in host.iter_mut().enumerate().take(start_pos + 1) {
            *slot = idx as i32;
        }
    }
    let data = ctx.stream.clone_htod(&host)?;
    ctx.sync()?;
    Ok((data, window_size))
}

pub fn compress_topk_indices(
    ctx: &RankGpuContext,
    seq_len: usize,
    ratio: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "seq_len must be positive");
    ensure!(ratio > 0, "compress ratio must be positive");
    let compressed = seq_len / ratio;
    ensure!(
        compressed > 0,
        "seq_len {seq_len} is smaller than ratio {ratio}"
    );
    let mut host = vec![-1_i32; seq_len * compressed];
    for token in 0..seq_len {
        let valid = (token + 1) / ratio;
        for block in 0..compressed {
            if block < valid {
                host[token * compressed + block] = (offset + block) as i32;
            }
        }
    }
    let data = ctx.stream.clone_htod(&host)?;
    ctx.sync()?;
    Ok((data, compressed))
}

pub fn compress_topk_indices_decode(
    ctx: &RankGpuContext,
    start_pos: usize,
    ratio: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(ratio > 0, "compress ratio must be positive");
    let compressed = (start_pos + 1) / ratio;
    let host = (0..compressed)
        .map(|block| (offset + block) as i32)
        .collect::<Vec<_>>();
    let data = ctx.stream.clone_htod(&host)?;
    ctx.sync()?;
    Ok((data, compressed))
}

pub fn window_and_compress_topk_indices(
    ctx: &RankGpuContext,
    seq_len: usize,
    window_size: usize,
    ratio: usize,
    compress_offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "seq_len must be positive");
    ensure!(window_size > 0, "window_size must be positive");
    ensure!(ratio > 0, "compress ratio must be positive");
    let window_topk = seq_len.min(window_size);
    let compressed = seq_len / ratio;
    ensure!(
        compressed > 0,
        "seq_len {seq_len} is smaller than ratio {ratio}"
    );
    let topk = window_topk + compressed;
    let mut host = vec![-1_i32; seq_len * topk];
    for token in 0..seq_len {
        let key_start = token.saturating_sub(window_size - 1);
        let mut route = 0;
        for key in key_start..=token {
            if route < window_topk {
                host[token * topk + route] = key as i32;
                route += 1;
            }
        }
        let valid = (token + 1) / ratio;
        for block in 0..compressed {
            if block < valid {
                host[token * topk + window_topk + block] = (compress_offset + block) as i32;
            }
        }
    }
    let data = ctx.stream.clone_htod(&host)?;
    ctx.sync()?;
    Ok((data, topk))
}

pub fn concat_seq_bf16_hidden(
    ctx: &RankGpuContext,
    a: &Bf16HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "concat hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len + b.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_concat_seq_bf16_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                a.seq_len as i32,
                b.seq_len as i32,
                a.hidden_dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn compressor_nonoverlap_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses the overlap compressor path");
    ensure!(
        start_pos == 0,
        "non-overlap compressor prefill currently supports start_pos=0 only"
    );
    ensure!(
        input.hidden_dim == config.dim,
        "compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    let compressed_len = input.seq_len / ratio;
    ensure!(
        compressed_len > 0,
        "input seq_len {} is smaller than ratio {}",
        input.seq_len,
        ratio
    );
    ensure!(
        compressor.ape.tensor.dtype == safetensors::Dtype::F32,
        "compressor ape {} must be F32, got {:?}",
        compressor.ape.name,
        compressor.ape.tensor.dtype
    );
    ensure!(
        compressor.wkv.tensor.dtype == safetensors::Dtype::BF16,
        "compressor wkv {} must be BF16, got {:?}",
        compressor.wkv.name,
        compressor.wkv.tensor.dtype
    );
    ensure!(
        compressor.wgate.tensor.dtype == safetensors::Dtype::BF16,
        "compressor wgate {} must be BF16, got {:?}",
        compressor.wgate.name,
        compressor.wgate.tensor.dtype
    );
    ensure!(
        compressor.norm.tensor.dtype == safetensors::Dtype::BF16,
        "compressor norm {} must be BF16, got {:?}",
        compressor.norm.name,
        compressor.norm.tensor.dtype
    );
    ensure!(
        compressor.ape.tensor.shape == [ratio, config.head_dim],
        "compressor ape {} shape mismatch: expected {:?}, got {:?}",
        compressor.ape.name,
        [ratio, config.head_dim],
        compressor.ape.tensor.shape
    );
    ensure!(
        compressor.wkv.tensor.shape == [config.head_dim, config.dim],
        "compressor wkv {} shape mismatch: expected {:?}, got {:?}",
        compressor.wkv.name,
        [config.head_dim, config.dim],
        compressor.wkv.tensor.shape
    );
    ensure!(
        compressor.wgate.tensor.shape == [config.head_dim, config.dim],
        "compressor wgate {} shape mismatch: expected {:?}, got {:?}",
        compressor.wgate.name,
        [config.head_dim, config.dim],
        compressor.wgate.tensor.shape
    );
    ensure!(
        compressor.norm.tensor.shape == [config.head_dim],
        "compressor norm {} shape mismatch: expected {:?}, got {:?}",
        compressor.norm.name,
        [config.head_dim],
        compressor.norm.tensor.shape
    );

    let mut weighted: CudaSlice<f32> = ctx.stream.alloc_zeros(compressed_len * config.head_dim)?;
    let mut out = Bf16HiddenStates::zeros(ctx, config.head_dim, compressed_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = weighted.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_prefill_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    apply_rope_hidden_strided_in_place(
        ctx,
        &mut out,
        rope,
        1,
        config.head_dim,
        start_pos,
        ratio,
        false,
    )?;
    fp8_act_quant_nope_bf16_hidden_in_place(
        ctx,
        &mut out,
        1,
        config.head_dim,
        rope.rotary_dim,
        64,
    )?;
    Ok(out)
}

pub fn compressor_overlap_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    compressor_overlap_prefill_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        compressor,
        rope,
        start_pos,
        config.head_dim,
    )
}

pub fn compressor_overlap_prefill_bf16_hidden_with_dim(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    head_dim: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "overlap compressor prefill currently supports start_pos=0 only"
    );
    ensure!(
        input.hidden_dim == config.dim,
        "overlap compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    let ratio = 4;
    let compressed_len = input.seq_len / ratio;
    ensure!(
        compressed_len > 0,
        "input seq_len {} is smaller than ratio {}",
        input.seq_len,
        ratio
    );
    ensure!(
        compressor.ape.tensor.dtype == safetensors::Dtype::F32,
        "overlap compressor ape {} must be F32, got {:?}",
        compressor.ape.name,
        compressor.ape.tensor.dtype
    );
    ensure!(
        compressor.wkv.tensor.dtype == safetensors::Dtype::BF16,
        "overlap compressor wkv {} must be BF16, got {:?}",
        compressor.wkv.name,
        compressor.wkv.tensor.dtype
    );
    ensure!(
        compressor.wgate.tensor.dtype == safetensors::Dtype::BF16,
        "overlap compressor wgate {} must be BF16, got {:?}",
        compressor.wgate.name,
        compressor.wgate.tensor.dtype
    );
    ensure!(
        compressor.norm.tensor.dtype == safetensors::Dtype::BF16,
        "overlap compressor norm {} must be BF16, got {:?}",
        compressor.norm.name,
        compressor.norm.tensor.dtype
    );
    ensure!(
        compressor.ape.tensor.shape == [ratio, 2 * head_dim],
        "overlap compressor ape {} shape mismatch: expected {:?}, got {:?}",
        compressor.ape.name,
        [ratio, 2 * head_dim],
        compressor.ape.tensor.shape
    );
    ensure!(
        compressor.wkv.tensor.shape == [2 * head_dim, config.dim],
        "overlap compressor wkv {} shape mismatch: expected {:?}, got {:?}",
        compressor.wkv.name,
        [2 * head_dim, config.dim],
        compressor.wkv.tensor.shape
    );
    ensure!(
        compressor.wgate.tensor.shape == [2 * head_dim, config.dim],
        "overlap compressor wgate {} shape mismatch: expected {:?}, got {:?}",
        compressor.wgate.name,
        [2 * head_dim, config.dim],
        compressor.wgate.tensor.shape
    );
    ensure!(
        compressor.norm.tensor.shape == [head_dim],
        "overlap compressor norm {} shape mismatch: expected {:?}, got {:?}",
        compressor.norm.name,
        [head_dim],
        compressor.norm.tensor.shape
    );

    let mut weighted: CudaSlice<f32> = ctx.stream.alloc_zeros(compressed_len * head_dim)?;
    let mut out = Bf16HiddenStates::zeros(ctx, head_dim, compressed_len)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = weighted.device_ptr_mut(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_compressor_overlap_prefill_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                weighted_ptr as *mut f32,
                out_ptr as *mut ffi::Half,
                input.seq_len as i32,
                input.hidden_dim as i32,
                head_dim as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    apply_rope_hidden_strided_in_place(ctx, &mut out, rope, 1, head_dim, start_pos, ratio, false)?;
    if head_dim == config.head_dim {
        fp8_act_quant_nope_bf16_hidden_in_place(ctx, &mut out, 1, head_dim, rope.rotary_dim, 64)?;
    }
    Ok(out)
}

pub fn compressor_nonoverlap_decode_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    state: &mut CompressorDecodeState,
) -> Result<Option<Bf16HiddenStates>> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses the overlap compressor path");
    ensure!(
        input.hidden_dim == config.dim,
        "decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "decode compressor expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        state.hidden_dim == config.head_dim && state.slots == ratio,
        "decode compressor state mismatch: hidden_dim={}, slots={}, expected {}x{}",
        state.hidden_dim,
        state.slots,
        config.head_dim,
        ratio
    );

    let should_compress = (start_pos + 1).is_multiple_of(ratio);
    let mut weighted = if should_compress {
        Some(ctx.stream.alloc_zeros::<f32>(config.head_dim)?)
    } else {
        None
    };
    let mut out = if should_compress {
        Some(Bf16HiddenStates::zeros(ctx, config.head_dim, 1)?)
    } else {
        None
    };
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = if let Some(weighted) = weighted.as_mut() {
            let (ptr, guard) = weighted.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let (out_ptr, _out_guard) = if let Some(out) = out.as_mut() {
            let (ptr, guard) = out.data.device_ptr_mut(&ctx.stream);
            (ptr as *mut ffi::Half, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let result = unsafe {
            ffi::deepseek_compressor_nonoverlap_decode_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr,
                out_ptr,
                start_pos as i32,
                input.hidden_dim as i32,
                config.head_dim as i32,
                ratio as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if let Some(mut out) = out {
        let rope_start = start_pos + 1 - ratio;
        apply_rope_hidden_strided_in_place(
            ctx,
            &mut out,
            rope,
            1,
            config.head_dim,
            rope_start,
            ratio,
            false,
        )?;
        fp8_act_quant_nope_bf16_hidden_in_place(
            ctx,
            &mut out,
            1,
            config.head_dim,
            rope.rotary_dim,
            64,
        )?;
        Ok(Some(out))
    } else {
        Ok(None)
    }
}

pub fn compressor_overlap_decode_bf16_hidden_with_dim(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    head_dim: usize,
    state: &mut CompressorDecodeState,
    rotate_fp4: bool,
) -> Result<Option<Bf16HiddenStates>> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "overlap decode compressor input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "overlap decode compressor expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        state.hidden_dim == 2 * head_dim && state.slots == 8,
        "overlap decode compressor state mismatch: hidden_dim={}, slots={}, expected {}x8",
        state.hidden_dim,
        state.slots,
        2 * head_dim
    );

    let should_compress = (start_pos + 1).is_multiple_of(4);
    let mut weighted = if should_compress {
        Some(ctx.stream.alloc_zeros::<f32>(head_dim)?)
    } else {
        None
    };
    let mut out = if should_compress {
        Some(Bf16HiddenStates::zeros(ctx, head_dim, 1)?)
    } else {
        None
    };
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (wkv_ptr, _wkv_guard) = compressor.wkv.tensor.data.device_ptr(&ctx.stream);
        let (wgate_ptr, _wgate_guard) = compressor.wgate.tensor.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ape_guard) = compressor.ape.tensor.data.device_ptr(&ctx.stream);
        let (norm_ptr, _norm_guard) = compressor.norm.tensor.data.device_ptr(&ctx.stream);
        let (kv_state_ptr, _kv_state_guard) = state.kv.device_ptr_mut(&ctx.stream);
        let (score_state_ptr, _score_state_guard) = state.score.device_ptr_mut(&ctx.stream);
        let (weighted_ptr, _weighted_guard) = if let Some(weighted) = weighted.as_mut() {
            let (ptr, guard) = weighted.device_ptr_mut(&ctx.stream);
            (ptr as *mut f32, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let (out_ptr, _out_guard) = if let Some(out) = out.as_mut() {
            let (ptr, guard) = out.data.device_ptr_mut(&ctx.stream);
            (ptr as *mut ffi::Half, Some(guard))
        } else {
            (ptr::null_mut(), None)
        };
        let result = unsafe {
            ffi::deepseek_compressor_overlap_decode_cuda(
                x_ptr as *const ffi::Half,
                wkv_ptr as *const ffi::Half,
                wgate_ptr as *const ffi::Half,
                ape_ptr as *const f32,
                norm_ptr as *const ffi::Half,
                kv_state_ptr as *mut f32,
                score_state_ptr as *mut f32,
                weighted_ptr,
                out_ptr,
                start_pos as i32,
                input.hidden_dim as i32,
                head_dim as i32,
                config.rms_norm_eps,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    if let Some(mut out) = out {
        let rope_start = start_pos + 1 - 4;
        apply_rope_hidden_strided_in_place(ctx, &mut out, rope, 1, head_dim, rope_start, 4, false)?;
        if rotate_fp4 {
            hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut out, 1, head_dim)?;
        } else {
            fp8_act_quant_nope_bf16_hidden_in_place(
                ctx,
                &mut out,
                1,
                head_dim,
                rope.rotary_dim,
                64,
            )?;
        }
        Ok(Some(out))
    } else {
        Ok(None)
    }
}

pub fn compressor_overlap_decode_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    state: &mut CompressorDecodeState,
) -> Result<Option<Bf16HiddenStates>> {
    ctx.set_current()?;
    compressor_overlap_decode_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        compressor,
        rope,
        start_pos,
        config.head_dim,
        state,
        false,
    )
}

pub(crate) fn init_nonoverlap_compressor_state_from_prefill(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    ratio: usize,
    rope: &DeepSeekRopeCache,
    state: &mut CompressorDecodeState,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(ratio > 1, "compress ratio must be > 1");
    ensure!(ratio != 4, "ratio=4 uses overlap compressor state");
    let tail = input.seq_len % ratio;
    if tail == 0 {
        return Ok(());
    }
    let start = input.seq_len - tail;
    for pos in start..input.seq_len {
        let row = copy_bf16_row_to_hidden(ctx, input, pos)?;
        let _ = compressor_nonoverlap_decode_bf16_hidden(
            ctx, config, &row, compressor, ratio, rope, pos, state,
        )?;
    }
    Ok(())
}

pub(crate) fn init_overlap_compressor_state_from_prefill(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    compressor: &CompressorWeights<'_>,
    rope: &DeepSeekRopeCache,
    head_dim: usize,
    state: &mut CompressorDecodeState,
    rotate_fp4: bool,
) -> Result<()> {
    ctx.set_current()?;
    let start = input.seq_len.saturating_sub(8);
    for pos in start..input.seq_len {
        let row = copy_bf16_row_to_hidden(ctx, input, pos)?;
        let _ = compressor_overlap_decode_bf16_hidden_with_dim(
            ctx, config, &row, compressor, rope, pos, head_dim, state, rotate_fp4,
        )?;
    }
    Ok(())
}

pub fn concat_topk_indices(
    ctx: &RankGpuContext,
    a: &CudaSlice<i32>,
    a_topk: usize,
    b: &CudaSlice<i32>,
    b_topk: usize,
    seq_len: usize,
) -> Result<CudaSlice<i32>> {
    ctx.set_current()?;
    ensure!(seq_len > 0, "top-k concat seq_len must be positive");
    ensure!(
        a.len() == seq_len * a_topk,
        "top-k concat left shape mismatch: expected {}, got {}",
        seq_len * a_topk,
        a.len()
    );
    ensure!(
        b.len() == seq_len * b_topk,
        "top-k concat right shape mismatch: expected {}, got {}",
        seq_len * b_topk,
        b.len()
    );
    let mut out = ctx.stream.alloc_zeros(seq_len * (a_topk + b_topk))?;
    {
        let (a_ptr, _a_guard) = a.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_concat_topk_indices_cuda(
                a_ptr as *const i32,
                b_ptr as *const i32,
                out_ptr as *mut i32,
                seq_len as i32,
                a_topk as i32,
                b_topk as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn hadamard_fp4_quant_bf16_hidden_in_place(
    ctx: &RankGpuContext,
    hidden: &mut Bf16HiddenStates,
    groups: usize,
    dim: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(groups > 0, "Hadamard groups must be positive");
    ensure!(dim > 0, "Hadamard dim must be positive");
    ensure!(
        hidden.hidden_dim == groups * dim,
        "Hadamard hidden dim mismatch: expected {}, got {}",
        groups * dim,
        hidden.hidden_dim
    );
    ensure!(
        dim.is_power_of_two(),
        "Hadamard dim must be a power of two, got {}",
        dim
    );
    ensure!(
        dim.is_multiple_of(32),
        "FP4 quant dim must be divisible by 32, got {}",
        dim
    );

    {
        let (x_ptr, _x_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hadamard_fp4_quant_bf16_cuda(
                x_ptr as *mut ffi::Half,
                hidden.seq_len as i32,
                groups as i32,
                dim as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn indexer_scores_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    qr: &Bf16HiddenStates,
    indexer: &IndexerWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<(CudaSlice<f32>, usize)> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "indexer prefill scores currently supports start_pos=0 only"
    );
    ensure!(
        input.hidden_dim == config.dim,
        "indexer input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        qr.hidden_dim == config.q_lora_rank,
        "indexer qr dim mismatch: expected {}, got {}",
        config.q_lora_rank,
        qr.hidden_dim
    );
    ensure!(
        qr.seq_len == input.seq_len,
        "indexer qr/input seq mismatch: qr={}, input={}",
        qr.seq_len,
        input.seq_len
    );
    ensure!(
        input.seq_len >= 4,
        "indexer prefill needs at least one ratio-4 block, got seq_len={}",
        input.seq_len
    );

    let local_heads = config.index_n_heads / 8;
    let mut q = fp8_linear_bf16_hidden(ctx, qr, &indexer.wq_b)?;
    ensure!(
        q.hidden_dim == local_heads * config.index_head_dim,
        "indexer q dim mismatch: expected {}, got {}",
        local_heads * config.index_head_dim,
        q.hidden_dim
    );
    apply_rope_hidden_in_place(
        ctx,
        &mut q,
        rope,
        local_heads,
        config.index_head_dim,
        start_pos,
        false,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut q, local_heads, config.index_head_dim)?;

    let mut compressed_kv = compressor_overlap_prefill_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        &indexer.compressor,
        rope,
        start_pos,
        config.index_head_dim,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut compressed_kv, 1, config.index_head_dim)?;
    let weights = bf16_linear_bf16_hidden(ctx, input, &indexer.weights_proj)?;
    ensure!(
        weights.hidden_dim == local_heads,
        "indexer weights dim mismatch: expected {}, got {}",
        local_heads,
        weights.hidden_dim
    );

    let compressed_len = compressed_kv.seq_len;
    let mut scores = ctx.stream.alloc_zeros(input.seq_len * compressed_len)?;
    {
        let (q_ptr, _q_guard) = q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = compressed_kv.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.data.device_ptr(&ctx.stream);
        let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
        let score_scale =
            1.0f32 / (config.index_head_dim as f32).sqrt() / (config.index_n_heads as f32).sqrt();
        let result = unsafe {
            ffi::deepseek_indexer_scores_prefill_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                weights_ptr as *const ffi::Half,
                scores_ptr as *mut f32,
                input.seq_len as i32,
                local_heads as i32,
                config.index_head_dim as i32,
                compressed_len as i32,
                score_scale,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((scores, compressed_len))
}

pub fn indexer_topk_indices_prefill(
    ctx: &RankGpuContext,
    config: &Config,
    scores: &CudaSlice<f32>,
    seq_len: usize,
    compressed_len: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(
        compressed_len > 0,
        "indexer compressed_len must be positive"
    );
    ensure!(
        scores.len() == seq_len * compressed_len,
        "indexer scores shape mismatch: expected {}, got {}",
        seq_len * compressed_len,
        scores.len()
    );
    let topk = config.index_topk.min(compressed_len);
    let mut topk_idxs = ctx.stream.alloc_zeros(seq_len * topk)?;
    {
        let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexer_topk_prefill_cuda(
                scores_ptr as *const f32,
                topk_ptr as *mut i32,
                seq_len as i32,
                compressed_len as i32,
                topk as i32,
                4,
                offset as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((topk_idxs, topk))
}

pub fn indexer_scores_decode_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    qr: &Bf16HiddenStates,
    indexer: &IndexerWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
    compressor_state: &mut CompressorDecodeState,
) -> Result<Option<CudaSlice<f32>>> {
    ctx.set_current()?;
    ensure!(
        input.hidden_dim == config.dim,
        "indexer decode input dim mismatch: expected {}, got {}",
        config.dim,
        input.hidden_dim
    );
    ensure!(
        input.seq_len == 1,
        "indexer decode expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        qr.hidden_dim == config.q_lora_rank && qr.seq_len == 1,
        "indexer decode qr shape mismatch: hidden_dim={}, seq_len={}",
        qr.hidden_dim,
        qr.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == config.index_head_dim,
        "indexer decode kv cache dim mismatch: expected {}, got {}",
        config.index_head_dim,
        kv_cache.hidden_dim
    );

    let local_heads = config.index_n_heads / 8;
    let mut q = fp8_linear_bf16_hidden(ctx, qr, &indexer.wq_b)?;
    ensure!(
        q.hidden_dim == local_heads * config.index_head_dim && q.seq_len == 1,
        "indexer decode q shape mismatch: hidden_dim={}, seq_len={}",
        q.hidden_dim,
        q.seq_len
    );
    apply_rope_hidden_in_place(
        ctx,
        &mut q,
        rope,
        local_heads,
        config.index_head_dim,
        start_pos,
        false,
    )?;
    hadamard_fp4_quant_bf16_hidden_in_place(ctx, &mut q, local_heads, config.index_head_dim)?;

    if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden_with_dim(
        ctx,
        config,
        input,
        &indexer.compressor,
        rope,
        start_pos,
        config.index_head_dim,
        compressor_state,
        true,
    )? {
        copy_bf16_rows_to_cache(ctx, &compressed_kv, kv_cache, 0, start_pos / 4, 1)?;
    }

    let compressed_len = (start_pos + 1) / 4;
    if compressed_len == 0 {
        return Ok(None);
    }
    ensure!(
        compressed_len <= kv_cache.slots,
        "indexer decode compressed_len {} exceeds kv cache slots {}",
        compressed_len,
        kv_cache.slots
    );

    let weights = bf16_linear_bf16_hidden(ctx, input, &indexer.weights_proj)?;
    ensure!(
        weights.hidden_dim == local_heads && weights.seq_len == 1,
        "indexer decode weights shape mismatch: hidden_dim={}, seq_len={}",
        weights.hidden_dim,
        weights.seq_len
    );
    let mut scores = ctx.stream.alloc_zeros(compressed_len)?;
    {
        let (q_ptr, _q_guard) = q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv_cache.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.data.device_ptr(&ctx.stream);
        let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
        let score_scale =
            1.0f32 / (config.index_head_dim as f32).sqrt() / (config.index_n_heads as f32).sqrt();
        let result = unsafe {
            ffi::deepseek_indexer_scores_decode_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                weights_ptr as *const ffi::Half,
                scores_ptr as *mut f32,
                local_heads as i32,
                config.index_head_dim as i32,
                compressed_len as i32,
                score_scale,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(Some(scores))
}

pub fn indexer_topk_indices_decode(
    ctx: &RankGpuContext,
    config: &Config,
    scores: &CudaSlice<f32>,
    compressed_len: usize,
    offset: usize,
) -> Result<(CudaSlice<i32>, usize)> {
    ctx.set_current()?;
    ensure!(
        compressed_len > 0,
        "indexer decode compressed_len must be positive"
    );
    ensure!(
        scores.len() == compressed_len,
        "indexer decode scores shape mismatch: expected {}, got {}",
        compressed_len,
        scores.len()
    );
    let topk = config.index_topk.min(compressed_len);
    let mut topk_idxs = ctx.stream.alloc_zeros(topk)?;
    {
        let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexer_topk_decode_cuda(
                scores_ptr as *const f32,
                topk_ptr as *mut i32,
                compressed_len as i32,
                topk as i32,
                offset as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok((topk_idxs, topk))
}

pub fn attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "compressed attention prefill currently supports start_pos=0 only"
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(ratio > 0, "layer {layer} is not compressed");
    ensure!(ratio != 4, "ratio=4 uses the indexer/overlap path");
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    if input.seq_len < ratio {
        let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
        return attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            projections.local_heads,
            projections.head_dim,
            start_pos,
        );
    }
    let compressed_kv = compressor_nonoverlap_prefill_bf16_hidden(
        ctx, config, input, compressor, ratio, rope, start_pos,
    )?;
    let kv = concat_seq_bf16_hidden(ctx, &projections.kv, &compressed_kv)?;
    let (topk_idxs, topk) = window_and_compress_topk_indices(
        ctx,
        projections.q.seq_len,
        config.sliding_window,
        ratio,
        projections.kv.seq_len,
    )?;
    let indexed_projections = AttentionProjections {
        qr: projections.qr,
        q: projections.q,
        kv,
        local_heads: projections.local_heads,
        head_dim: projections.head_dim,
    };
    let mut attn_out = indexed_attention_prefill_bf16_hidden(
        ctx,
        config,
        &indexed_projections,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        indexed_projections.local_heads,
        indexed_projections.head_dim,
        start_pos,
    )
}

pub(crate) fn attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden_with_cache(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "compressed attention prefill cache path currently supports start_pos=0 only"
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(ratio > 0, "layer {layer} is not compressed");
    ensure!(ratio != 4, "ratio=4 uses the indexer/overlap path");
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor decode state"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_window_prefill_to_ring_cache(ctx, &projections.kv, &mut cache.kv, config.sliding_window)?;
    init_nonoverlap_compressor_state_from_prefill(
        ctx,
        config,
        input,
        compressor,
        ratio,
        rope,
        compressor_state,
    )?;
    if input.seq_len < ratio {
        let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
        return attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            projections.local_heads,
            projections.head_dim,
            start_pos,
        );
    }
    let compressed_kv = compressor_nonoverlap_prefill_bf16_hidden(
        ctx, config, input, compressor, ratio, rope, start_pos,
    )?;
    copy_bf16_rows_to_cache(
        ctx,
        &compressed_kv,
        &mut cache.kv,
        0,
        config.sliding_window,
        compressed_kv.seq_len,
    )?;
    let kv = concat_seq_bf16_hidden(ctx, &projections.kv, &compressed_kv)?;
    let (topk_idxs, topk) = window_and_compress_topk_indices(
        ctx,
        projections.q.seq_len,
        config.sliding_window,
        ratio,
        projections.kv.seq_len,
    )?;
    let indexed_projections = AttentionProjections {
        qr: projections.qr,
        q: projections.q,
        kv,
        local_heads: projections.local_heads,
        head_dim: projections.head_dim,
    };
    let mut attn_out = indexed_attention_prefill_bf16_hidden(
        ctx,
        config,
        &indexed_projections,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        indexed_projections.local_heads,
        indexed_projections.head_dim,
        start_pos,
    )
}

pub fn attention_prefill_compressed_nonoverlap_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "compressed attention group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
        out.push(
            attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden(
                ctx, config, input, attn, rope, layer, start_pos,
            )?,
        );
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_prefill_compressed_nonoverlap_group_bf16_hidden_with_cache(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "compressed attention cache group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        out.push(
            attention_prefill_compressed_nonoverlap_rank_local_bf16_hidden_with_cache(
                ctx, config, input, attn, rope, layer, start_pos, cache,
            )?,
        );
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

fn finish_compressed_overlap_attention_rank_local(
    ctx: &RankGpuContext,
    config: &Config,
    projections: AttentionProjections,
    compressed_kv: Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    let kv = concat_seq_bf16_hidden(ctx, &projections.kv, &compressed_kv)?;
    let indexed_projections = AttentionProjections {
        qr: projections.qr,
        q: projections.q,
        kv,
        local_heads: projections.local_heads,
        head_dim: projections.head_dim,
    };
    let mut attn_out = indexed_attention_prefill_bf16_hidden(
        ctx,
        config,
        &indexed_projections,
        attn,
        topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        indexed_projections.local_heads,
        indexed_projections.head_dim,
        start_pos,
    )
}

pub fn attention_prefill_compressed_overlap_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    layer: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        start_pos == 0,
        "ratio-4 overlap attention prefill currently supports start_pos=0 only"
    );
    ensure!(
        layer < config.compress_ratios.len(),
        "layer {layer} out of range"
    );
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 overlap attention called for layer {layer} with compress_ratio={}",
        config.compress_ratios[layer]
    );
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    if input.seq_len < 4 {
        let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
        return attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            projections.local_heads,
            projections.head_dim,
            start_pos,
        );
    }
    let compressed_kv =
        compressor_overlap_prefill_bf16_hidden(ctx, config, input, compressor, rope, start_pos)?;
    let (topk_idxs, topk) = window_and_compress_topk_indices(
        ctx,
        projections.q.seq_len,
        config.sliding_window,
        4,
        projections.kv.seq_len,
    )?;
    finish_compressed_overlap_attention_rank_local(
        ctx,
        config,
        projections,
        compressed_kv,
        attn,
        rope,
        &topk_idxs,
        topk,
        start_pos,
    )
}

pub fn attention_prefill_compressed_overlap_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "ratio-4 attention group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    if ranks[0].3.seq_len < 4 {
        let mut out = Vec::with_capacity(ranks.len());
        for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
            out.push(attention_prefill_compressed_overlap_rank_local_bf16_hidden(
                ctx, config, input, attn, rope, layer, start_pos,
            )?);
        }
        let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
            .iter()
            .zip(out.iter_mut())
            .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
            .collect();
        all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
        return Ok(out);
    }

    let mut projections = Vec::with_capacity(ranks.len());
    let mut compressed_kvs = Vec::with_capacity(ranks.len());
    let mut index_scores = Vec::with_capacity(ranks.len());
    let mut compressed_len = None;
    for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
        ensure!(
            config.compress_ratios[layer] == 4,
            "ratio-4 group attention called for layer {layer} with compress_ratio={}",
            config.compress_ratios[layer]
        );
        let compressor = attn
            .compressor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;
        let indexer = attn
            .indexer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing ratio-4 indexer weights"))?;

        let mut rank_projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
        apply_rope_attention_projections(ctx, &mut rank_projections, rope, start_pos)?;
        let rank_compressed_kv = compressor_overlap_prefill_bf16_hidden(
            ctx, config, input, compressor, rope, start_pos,
        )?;
        let (rank_scores, rank_compressed_len) = indexer_scores_prefill_bf16_hidden(
            ctx,
            config,
            input,
            &rank_projections.qr,
            indexer,
            rope,
            start_pos,
        )?;
        if let Some(expected) = compressed_len {
            ensure!(
                rank_compressed_len == expected,
                "indexer compressed len mismatch: expected {}, got {}",
                expected,
                rank_compressed_len
            );
        } else {
            compressed_len = Some(rank_compressed_len);
        }

        projections.push(rank_projections);
        compressed_kvs.push(rank_compressed_kv);
        index_scores.push(rank_scores);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _), scores) in ranks.iter().zip(index_scores.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(scores, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!(
                "NCCL indexer score all-reduce failed: {err:?}"
            ));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    let compressed_len =
        compressed_len.ok_or_else(|| anyhow::anyhow!("ratio-4 group has no compressed len"))?;
    let mut out = Vec::with_capacity(ranks.len());
    for (((((ctx, attn, _comm, input), rope), rank_projections), rank_compressed_kv), scores) in
        ranks
            .iter()
            .zip(ropes.iter())
            .zip(projections.into_iter())
            .zip(compressed_kvs.into_iter())
            .zip(index_scores.iter())
    {
        let (window_idxs, window_topk) =
            window_topk_indices(ctx, input.seq_len, config.sliding_window)?;
        let (compress_idxs, compress_topk) = indexer_topk_indices_prefill(
            ctx,
            config,
            scores,
            input.seq_len,
            compressed_len,
            rank_projections.kv.seq_len,
        )?;
        let topk_idxs = concat_topk_indices(
            ctx,
            &window_idxs,
            window_topk,
            &compress_idxs,
            compress_topk,
            input.seq_len,
        )?;
        out.push(finish_compressed_overlap_attention_rank_local(
            ctx,
            config,
            rank_projections,
            rank_compressed_kv,
            attn,
            rope,
            &topk_idxs,
            window_topk + compress_topk,
            start_pos,
        )?);
    }

    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_prefill_compressed_overlap_group_bf16_hidden_with_cache(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "ratio-4 attention cache group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );

    let mut projections = Vec::with_capacity(ranks.len());
    let mut compressed_kvs = Vec::with_capacity(ranks.len());
    let mut index_scores = Vec::with_capacity(ranks.len());
    let mut compressed_len = None;
    for (((ctx, attn, _comm, input, cache), rope), rank) in
        ranks.iter_mut().zip(ropes.iter()).zip(0..)
    {
        ensure!(
            config.compress_ratios[layer] == 4,
            "ratio-4 group attention cache path called for layer {layer} with compress_ratio={}",
            config.compress_ratios[layer]
        );
        let compressor = attn
            .compressor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;
        let indexer = attn
            .indexer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing ratio-4 indexer weights"))?;
        let compressor_state = cache
            .compressor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor state"))?;
        let indexer_kv = cache
            .indexer_kv
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer kv cache"))?;
        let indexer_state = cache
            .indexer_compressor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer compressor state"))?;

        let mut rank_projections = attention_project_bf16_hidden(ctx, config, input, attn)
            .with_context(|| {
                format!("ratio-4 cache attention_project layer {layer} rank {rank}")
            })?;
        apply_rope_attention_projections(ctx, &mut rank_projections, rope, start_pos)
            .with_context(|| format!("ratio-4 cache apply_rope layer {layer} rank {rank}"))?;
        copy_window_prefill_to_ring_cache(
            ctx,
            &rank_projections.kv,
            &mut cache.kv,
            config.sliding_window,
        )
        .with_context(|| format!("ratio-4 cache raw kv layer {layer} rank {rank}"))?;
        init_overlap_compressor_state_from_prefill(
            ctx,
            config,
            input,
            compressor,
            rope,
            config.head_dim,
            compressor_state,
            false,
        )
        .with_context(|| format!("ratio-4 cache compressor tail layer {layer} rank {rank}"))?;
        init_overlap_compressor_state_from_prefill(
            ctx,
            config,
            input,
            &indexer.compressor,
            rope,
            config.index_head_dim,
            indexer_state,
            true,
        )
        .with_context(|| format!("ratio-4 cache indexer tail layer {layer} rank {rank}"))?;

        if input.seq_len < 4 {
            projections.push(rank_projections);
            compressed_kvs.push(None);
            index_scores.push(None);
            continue;
        }

        let rank_compressed_kv = compressor_overlap_prefill_bf16_hidden(
            ctx, config, input, compressor, rope, start_pos,
        )?;
        copy_bf16_rows_to_cache(
            ctx,
            &rank_compressed_kv,
            &mut cache.kv,
            0,
            config.sliding_window,
            rank_compressed_kv.seq_len,
        )
        .with_context(|| format!("ratio-4 cache compressed kv layer {layer} rank {rank}"))?;
        let indexer_compressed_kv = compressor_overlap_prefill_bf16_hidden_with_dim(
            ctx,
            config,
            input,
            &indexer.compressor,
            rope,
            start_pos,
            config.index_head_dim,
        )?;
        copy_bf16_rows_to_cache(
            ctx,
            &indexer_compressed_kv,
            indexer_kv,
            0,
            0,
            indexer_compressed_kv.seq_len,
        )
        .with_context(|| format!("ratio-4 cache indexer kv layer {layer} rank {rank}"))?;
        let (rank_scores, rank_compressed_len) = indexer_scores_prefill_bf16_hidden(
            ctx,
            config,
            input,
            &rank_projections.qr,
            indexer,
            rope,
            start_pos,
        )?;
        if let Some(expected) = compressed_len {
            ensure!(
                rank_compressed_len == expected,
                "indexer compressed len mismatch: expected {}, got {}",
                expected,
                rank_compressed_len
            );
        } else {
            compressed_len = Some(rank_compressed_len);
        }

        projections.push(rank_projections);
        compressed_kvs.push(Some(rank_compressed_kv));
        index_scores.push(Some(rank_scores));
    }

    if let Some(_compressed_len) = compressed_len {
        group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
        for ((_, _, comm, _, _), scores) in ranks.iter().zip(index_scores.iter_mut()) {
            let Some(scores) = scores.as_mut() else {
                let _ = group_end();
                return Err(anyhow::anyhow!("missing rank indexer scores"));
            };
            if let Err(err) = comm.all_reduce_in_place(scores, &ReduceOp::Sum) {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "NCCL indexer score all-reduce failed: {err:?}"
                ));
            }
        }
        group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    }

    let mut out = Vec::with_capacity(ranks.len());
    for (
        ((((ctx, attn, _comm, input, _cache), rope), rank_projections), rank_compressed_kv),
        scores,
    ) in ranks
        .iter()
        .zip(ropes.iter())
        .zip(projections.into_iter())
        .zip(compressed_kvs.into_iter())
        .zip(index_scores.iter())
    {
        if input.seq_len < 4 {
            let mut attn_out =
                sparse_attention_prefill_bf16_hidden(ctx, config, &rank_projections, attn)?;
            out.push(attention_output_project_bf16_hidden(
                ctx,
                &mut attn_out,
                attn,
                rope,
                rank_projections.local_heads,
                rank_projections.head_dim,
                start_pos,
            )?);
            continue;
        }
        let compressed_len =
            compressed_len.ok_or_else(|| anyhow::anyhow!("ratio-4 group has no compressed len"))?;
        let rank_compressed_kv =
            rank_compressed_kv.ok_or_else(|| anyhow::anyhow!("missing compressed kv"))?;
        let scores = scores
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing indexer scores"))?;
        let (window_idxs, window_topk) =
            window_topk_indices(ctx, input.seq_len, config.sliding_window)?;
        let (compress_idxs, compress_topk) = indexer_topk_indices_prefill(
            ctx,
            config,
            scores,
            input.seq_len,
            compressed_len,
            rank_projections.kv.seq_len,
        )?;
        let topk_idxs = concat_topk_indices(
            ctx,
            &window_idxs,
            window_topk,
            &compress_idxs,
            compress_topk,
            input.seq_len,
        )?;
        out.push(finish_compressed_overlap_attention_rank_local(
            ctx,
            config,
            rank_projections,
            rank_compressed_kv,
            attn,
            rope,
            &topk_idxs,
            window_topk + compress_topk,
            start_pos,
        )?);
    }

    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub fn indexed_attention_prefill_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(topk > 0, "indexed attention topk must be positive");
    ensure!(
        topk_idxs.len() == projections.q.seq_len * topk,
        "indexed attention topk shape mismatch: expected {}, got {}",
        projections.q.seq_len * topk,
        topk_idxs.len()
    );
    ensure!(
        attn.attn_sink.tensor.dtype == safetensors::Dtype::F32,
        "attn_sink {} must be F32, got {:?}",
        attn.attn_sink.name,
        attn.attn_sink.tensor.dtype
    );
    ensure!(
        attn.attn_sink.tensor.shape == [projections.local_heads],
        "attn_sink {} shape mismatch: expected {:?}, got {:?}",
        attn.attn_sink.name,
        [projections.local_heads],
        attn.attn_sink.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, projections.q.hidden_dim, projections.q.seq_len)?;
    {
        let (q_ptr, _q_guard) = projections.q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = projections.kv.data.device_ptr(&ctx.stream);
        let (sink_ptr, _sink_guard) = attn.attn_sink.tensor.data.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexed_attention_prefill_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                sink_ptr as *const f32,
                topk_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                projections.q.seq_len as i32,
                projections.kv.seq_len as i32,
                projections.local_heads as i32,
                projections.head_dim as i32,
                topk as i32,
                1.0f32 / (config.head_dim as f32).sqrt(),
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn indexed_attention_cache_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    projections: &AttentionProjections,
    kv_cache: &Bf16Cache,
    attn: &AttentionWeights<'_>,
    topk_idxs: &CudaSlice<i32>,
    topk: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(topk > 0, "indexed cache attention topk must be positive");
    ensure!(
        projections.q.seq_len == 1,
        "indexed cache attention currently expects decode seq_len=1, got {}",
        projections.q.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == projections.head_dim,
        "kv cache hidden dim mismatch: expected {}, got {}",
        projections.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        topk_idxs.len() == topk,
        "indexed cache attention topk shape mismatch: expected {}, got {}",
        topk,
        topk_idxs.len()
    );
    ensure!(
        attn.attn_sink.tensor.dtype == safetensors::Dtype::F32,
        "attn_sink {} must be F32, got {:?}",
        attn.attn_sink.name,
        attn.attn_sink.tensor.dtype
    );
    ensure!(
        attn.attn_sink.tensor.shape == [projections.local_heads],
        "attn_sink {} shape mismatch: expected {:?}, got {:?}",
        attn.attn_sink.name,
        [projections.local_heads],
        attn.attn_sink.tensor.shape
    );

    let mut out = Bf16HiddenStates::zeros(ctx, projections.q.hidden_dim, projections.q.seq_len)?;
    {
        let (q_ptr, _q_guard) = projections.q.data.device_ptr(&ctx.stream);
        let (kv_ptr, _kv_guard) = kv_cache.data.device_ptr(&ctx.stream);
        let (sink_ptr, _sink_guard) = attn.attn_sink.tensor.data.device_ptr(&ctx.stream);
        let (topk_ptr, _topk_guard) = topk_idxs.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_indexed_attention_prefill_cuda(
                q_ptr as *const ffi::Half,
                kv_ptr as *const ffi::Half,
                sink_ptr as *const f32,
                topk_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                projections.q.seq_len as i32,
                kv_cache.slots as i32,
                projections.local_heads as i32,
                projections.head_dim as i32,
                topk as i32,
                1.0f32 / (config.head_dim as f32).sqrt(),
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn attention_output_project_bf16_hidden(
    ctx: &RankGpuContext,
    attn_out: &mut Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    local_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    apply_rope_hidden_in_place(ctx, attn_out, rope, local_heads, head_dim, start_pos, true)?;
    let low_rank = bf16_linear_bf16_hidden(ctx, attn_out, &attn.wo_a)?;
    fp8_linear_bf16_hidden(ctx, &low_rank, &attn.wo_b)
}

pub fn attention_prefill_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local prefill attention currently only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        start_pos == 0,
        "rank-local prefill attention currently supports start_pos=0 only, got {start_pos}"
    );
    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

pub(crate) fn attention_prefill_rank_local_bf16_hidden_with_cache(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local prefill attention cache path only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        start_pos == 0,
        "rank-local prefill attention cache path currently supports start_pos=0 only, got {start_pos}"
    );
    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_window_prefill_to_ring_cache(ctx, &projections.kv, kv_cache, config.sliding_window)?;
    let mut attn_out = sparse_attention_prefill_bf16_hidden(ctx, config, &projections, attn)?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

pub fn attention_prefill_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "attention group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input), rope) in ranks.iter().zip(ropes.iter()) {
        out.push(attention_prefill_rank_local_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos,
        )?);
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub fn attention_decode_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    kv_cache: &mut Bf16Cache,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        config.compress_ratios[layer] == 0,
        "rank-local decode attention currently only supports non-compressed layers, layer {layer} has compress_ratio={}",
        config.compress_ratios[layer]
    );
    ensure!(
        input.seq_len == 1,
        "rank-local decode attention expects seq_len=1, got {}",
        input.seq_len
    );
    ensure!(
        kv_cache.hidden_dim == config.head_dim,
        "decode kv cache hidden dim mismatch: expected {}, got {}",
        config.head_dim,
        kv_cache.hidden_dim
    );
    ensure!(
        kv_cache.slots >= config.sliding_window,
        "decode kv cache slots {} smaller than sliding_window {}",
        kv_cache.slots,
        config.sliding_window
    );

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)
        .with_context(|| format!("attention_project layer {layer}"))?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)
        .with_context(|| format!("apply_rope_attention_projections layer {layer}"))?;
    copy_bf16_rows_to_cache(
        ctx,
        &projections.kv,
        kv_cache,
        0,
        start_pos % config.sliding_window,
        1,
    )
    .with_context(|| format!("copy kv to cache layer {layer} pos {start_pos}"))?;
    let (topk_idxs, topk) = window_topk_indices_decode(ctx, start_pos, config.sliding_window)
        .with_context(|| format!("window_topk_indices_decode layer {layer} pos {start_pos}"))?;
    let mut attn_out = indexed_attention_cache_bf16_hidden(
        ctx,
        config,
        &projections,
        kv_cache,
        attn,
        &topk_idxs,
        topk,
    )
    .with_context(|| format!("indexed_attention_cache layer {layer} topk {topk}"))?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
    .with_context(|| format!("attention_output_project layer {layer}"))
}

pub fn attention_decode_group_bf16_hidden(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut Bf16Cache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "attention decode group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, kv_cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        out.push(attention_decode_rank_local_bf16_hidden(
            ctx, config, layer, input, attn, rope, start_pos, kv_cache,
        )?);
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    layer: usize,
    input: &Bf16HiddenStates,
    attn: &AttentionWeights<'_>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
    cache: &mut LayerDecodeCache,
) -> Result<Bf16HiddenStates> {
    ensure!(input.seq_len == 1, "compressed decode expects seq_len=1");
    ensure!(
        layer < config.compress_ratios.len(),
        "compressed decode layer {layer} out of range"
    );
    let ratio = config.compress_ratios[layer];
    ensure!(
        ratio > 0 && ratio != 4,
        "non-overlap decode called for ratio {ratio}"
    );
    ensure!(
        cache.kv.hidden_dim == config.head_dim && cache.kv.slots >= config.sliding_window,
        "compressed decode kv cache shape mismatch: hidden_dim={}, slots={}",
        cache.kv.hidden_dim,
        cache.kv.slots
    );
    let compressor = attn
        .compressor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor weights"))?;
    let compressor_state = cache
        .compressor
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("layer {layer} missing compressor decode state"))?;

    let mut projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
    apply_rope_attention_projections(ctx, &mut projections, rope, start_pos)?;
    copy_bf16_rows_to_cache(
        ctx,
        &projections.kv,
        &mut cache.kv,
        0,
        start_pos % config.sliding_window,
        1,
    )?;
    if let Some(compressed_kv) = compressor_nonoverlap_decode_bf16_hidden(
        ctx,
        config,
        input,
        compressor,
        ratio,
        rope,
        start_pos,
        compressor_state,
    )? {
        copy_bf16_rows_to_cache(
            ctx,
            &compressed_kv,
            &mut cache.kv,
            0,
            config.sliding_window + start_pos / ratio,
            1,
        )?;
    }

    let (window_idxs, window_topk) =
        window_topk_indices_decode(ctx, start_pos, config.sliding_window)?;
    let compressed_len = (start_pos + 1) / ratio;
    let (topk_idxs, topk) = if compressed_len > 0 {
        let (compress_idxs, compress_topk) =
            compress_topk_indices_decode(ctx, start_pos, ratio, config.sliding_window)?;
        (
            concat_topk_indices(
                ctx,
                &window_idxs,
                window_topk,
                &compress_idxs,
                compress_topk,
                1,
            )?,
            window_topk + compress_topk,
        )
    } else {
        (window_idxs, window_topk)
    };
    let mut attn_out = indexed_attention_cache_bf16_hidden(
        ctx,
        config,
        &projections,
        &cache.kv,
        attn,
        &topk_idxs,
        topk,
    )?;
    attention_output_project_bf16_hidden(
        ctx,
        &mut attn_out,
        attn,
        rope,
        projections.local_heads,
        projections.head_dim,
        start_pos,
    )
}

pub(crate) fn attention_decode_compressed_nonoverlap_group_bf16_hidden(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "compressed decode group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        out.push(
            attention_decode_compressed_nonoverlap_rank_local_bf16_hidden(
                ctx, config, layer, input, attn, rope, start_pos, cache,
            )?,
        );
    }
    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub(crate) fn attention_decode_compressed_overlap_group_bf16_hidden(
    ranks: &mut [(
        &RankGpuContext,
        &AttentionWeights<'_>,
        &Comm,
        &Bf16HiddenStates,
        &mut LayerDecodeCache,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        ranks.len() == ropes.len(),
        "ratio-4 decode group ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        config.compress_ratios[layer] == 4,
        "ratio-4 decode called for layer {layer} with ratio {}",
        config.compress_ratios[layer]
    );

    let mut projections = Vec::with_capacity(ranks.len());
    let mut index_scores: Vec<Option<CudaSlice<f32>>> = Vec::with_capacity(ranks.len());
    for ((ctx, attn, _comm, input, cache), rope) in ranks.iter_mut().zip(ropes.iter()) {
        ensure!(input.seq_len == 1, "ratio-4 decode expects seq_len=1");
        let compressor = attn
            .compressor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor weights"))?;
        let indexer = attn
            .indexer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing ratio-4 indexer weights"))?;
        let compressor_state = cache
            .compressor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing overlap compressor state"))?;

        let mut rank_projections = attention_project_bf16_hidden(ctx, config, input, attn)?;
        apply_rope_attention_projections(ctx, &mut rank_projections, rope, start_pos)?;
        copy_bf16_rows_to_cache(
            ctx,
            &rank_projections.kv,
            &mut cache.kv,
            0,
            start_pos % config.sliding_window,
            1,
        )?;
        if let Some(compressed_kv) = compressor_overlap_decode_bf16_hidden(
            ctx,
            config,
            input,
            compressor,
            rope,
            start_pos,
            compressor_state,
        )? {
            copy_bf16_rows_to_cache(
                ctx,
                &compressed_kv,
                &mut cache.kv,
                0,
                config.sliding_window + start_pos / 4,
                1,
            )?;
        }

        let indexer_kv = cache
            .indexer_kv
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer kv cache"))?;
        let indexer_state = cache
            .indexer_compressor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing indexer compressor state"))?;
        let scores = indexer_scores_decode_bf16_hidden(
            ctx,
            config,
            input,
            &rank_projections.qr,
            indexer,
            rope,
            start_pos,
            indexer_kv,
            indexer_state,
        )?;
        projections.push(rank_projections);
        index_scores.push(scores);
    }

    if let Some(compressed_len) = index_scores
        .iter()
        .find_map(|scores| scores.as_ref().map(|scores| scores.len()))
    {
        group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
        for ((_, _, comm, _, _), scores) in ranks.iter().zip(index_scores.iter_mut()) {
            let Some(scores) = scores.as_mut() else {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "missing rank indexer scores for compressed_len={compressed_len}"
                ));
            };
            if scores.len() != compressed_len {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "indexer decode score len mismatch: expected {}, got {}",
                    compressed_len,
                    scores.len()
                ));
            }
            if let Err(err) = comm.all_reduce_in_place(scores, &ReduceOp::Sum) {
                let _ = group_end();
                return Err(anyhow::anyhow!(
                    "NCCL decode indexer score all-reduce failed: {err:?}"
                ));
            }
        }
        group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    }

    let mut out = Vec::with_capacity(ranks.len());
    for ((((ctx, attn, _comm, _input, cache), rope), rank_projections), scores) in ranks
        .iter_mut()
        .zip(ropes.iter())
        .zip(projections.into_iter())
        .zip(index_scores.iter())
    {
        let (window_idxs, window_topk) =
            window_topk_indices_decode(ctx, start_pos, config.sliding_window)?;
        let compressed_len = (start_pos + 1) / 4;
        let (topk_idxs, topk) = if compressed_len > 0 {
            let scores = scores
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing indexer decode scores"))?;
            let (compress_idxs, compress_topk) = indexer_topk_indices_decode(
                ctx,
                config,
                scores,
                compressed_len,
                config.sliding_window,
            )?;
            (
                concat_topk_indices(
                    ctx,
                    &window_idxs,
                    window_topk,
                    &compress_idxs,
                    compress_topk,
                    1,
                )?,
                window_topk + compress_topk,
            )
        } else {
            (window_idxs, window_topk)
        };
        let mut attn_out = indexed_attention_cache_bf16_hidden(
            ctx,
            config,
            &rank_projections,
            &cache.kv,
            attn,
            &topk_idxs,
            topk,
        )?;
        out.push(attention_output_project_bf16_hidden(
            ctx,
            &mut attn_out,
            attn,
            rope,
            rank_projections.local_heads,
            rank_projections.head_dim,
            start_pos,
        )?);
    }

    let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> = ranks
        .iter()
        .zip(out.iter_mut())
        .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
        .collect();
    all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
    Ok(out)
}

pub fn block_prefill_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &HcHiddenStates,
    token_ids: &CudaSlice<u32>,
    rope: &DeepSeekRopeCache,
    start_pos: usize,
) -> Result<HcHiddenStates> {
    ctx.set_current()?;
    let block = weights.block(layer)?;

    let (attn_input, attn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        input,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
    )?;
    let attn_norm = rms_norm_bf16_hidden(ctx, &attn_input, &block.attn_norm, config.rms_norm_eps)?;
    let attn_out = attention_prefill_rank_local_bf16_hidden(
        ctx,
        config,
        layer,
        &attn_norm,
        &block.attn,
        rope,
        start_pos,
    )?;
    let after_attn = hc_post_bf16_hidden(ctx, &attn_out, input, &attn_hc)?;

    let (ffn_input, ffn_hc) = hc_pre_bf16_hidden(
        ctx,
        config,
        &after_attn,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
    )?;
    let ffn_norm = rms_norm_bf16_hidden(ctx, &ffn_input, &block.ffn_norm, config.rms_norm_eps)?;
    let ffn_out = moe_rank_local_bf16_hidden(ctx, config, weights, layer, &ffn_norm, token_ids)?;
    hc_post_bf16_hidden(ctx, &ffn_out, &after_attn, &ffn_hc)
}

pub fn block_prefill_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &HcHiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
) -> Result<Vec<HcHiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "block prefill group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "block prefill ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        layer < config.n_layers,
        "group block prefill layer {layer} out of range"
    );

    let blocks = ranks
        .iter()
        .enumerate()
        .map(|(rank, (_, weights, _, _, _))| {
            weights
                .block(layer)
                .with_context(|| format!("load block view layer {layer} rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut attn_inputs = Vec::with_capacity(ranks.len());
    let mut attn_hc = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, input, _), block) in ranks.iter().zip(blocks.iter()) {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_attn_fn,
            &block.hc_attn_scale,
            &block.hc_attn_base,
        )?;
        attn_inputs.push(pre);
        attn_hc.push(state);
    }

    let mut attn_norms = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, attn_input)) in
        ranks.iter().zip(blocks.iter().zip(attn_inputs.iter()))
    {
        attn_norms.push(rms_norm_bf16_hidden(
            ctx,
            attn_input,
            &block.attn_norm,
            config.rms_norm_eps,
        )?);
    }

    let attention_group = ranks
        .iter()
        .zip(blocks.iter())
        .zip(attn_norms.iter())
        .map(|(((ctx, _, comm, _, _), block), attn_norm)| (*ctx, &block.attn, *comm, attn_norm))
        .collect::<Vec<_>>();
    let attn_out = match config.compress_ratios[layer] {
        0 => {
            attention_prefill_group_bf16_hidden(&attention_group, config, layer, ropes, start_pos)?
        }
        4 => attention_prefill_compressed_overlap_group_bf16_hidden(
            &attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
        _ => attention_prefill_compressed_nonoverlap_group_bf16_hidden(
            &attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
    };

    let mut after_attn = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, input, _), attn_out), state)) in ranks
        .iter()
        .zip(attn_out.iter())
        .zip(attn_hc.iter())
        .enumerate()
    {
        after_attn.push(
            hc_post_bf16_hidden(ctx, attn_out, input, state)
                .with_context(|| format!("hc_post attention layer {layer} rank {rank}"))?,
        );
    }

    let mut ffn_inputs = Vec::with_capacity(ranks.len());
    let mut ffn_hc = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, input))) in ranks
        .iter()
        .zip(blocks.iter().zip(after_attn.iter()))
        .enumerate()
    {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_ffn_fn,
            &block.hc_ffn_scale,
            &block.hc_ffn_base,
        )
        .with_context(|| format!("hc_pre ffn layer {layer} rank {rank}"))?;
        ffn_inputs.push(pre);
        ffn_hc.push(state);
    }

    let mut ffn_norms = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, ffn_input))) in ranks
        .iter()
        .zip(blocks.iter().zip(ffn_inputs.iter()))
        .enumerate()
    {
        ffn_norms.push(
            rms_norm_bf16_hidden(ctx, ffn_input, &block.ffn_norm, config.rms_norm_eps)
                .with_context(|| format!("ffn rms_norm layer {layer} rank {rank}"))?,
        );
    }

    let moe_group = ranks
        .iter()
        .zip(ffn_norms.iter())
        .map(|((ctx, weights, comm, _, token_ids), ffn_norm)| {
            (*ctx, *weights, *comm, ffn_norm, *token_ids)
        })
        .collect::<Vec<_>>();
    let ffn_out = moe_group_bf16_hidden(&moe_group, config, layer)
        .with_context(|| format!("moe_group_bf16_hidden layer {layer}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, _, _), ffn_out), (input, state))) in ranks
        .iter()
        .zip(ffn_out.iter())
        .zip(after_attn.iter().zip(ffn_hc.iter()))
        .enumerate()
    {
        out.push(
            hc_post_bf16_hidden(ctx, ffn_out, input, state)
                .with_context(|| format!("hc_post ffn layer {layer} rank {rank}"))?,
        );
    }
    Ok(out)
}

pub(crate) fn block_prefill_group_bf16_hidden_with_decode_cache(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &HcHiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
    caches: &mut [LayerDecodeCache],
) -> Result<Vec<HcHiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "block prefill cache group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "block prefill cache ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        ranks.len() == caches.len(),
        "block prefill cache ranks/cache length mismatch: ranks={}, caches={}",
        ranks.len(),
        caches.len()
    );
    ensure!(
        layer < config.n_layers,
        "group block prefill cache layer {layer} out of range"
    );

    let blocks = ranks
        .iter()
        .enumerate()
        .map(|(rank, (_, weights, _, _, _))| {
            weights
                .block(layer)
                .with_context(|| format!("load block view layer {layer} rank {rank}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut attn_inputs = Vec::with_capacity(ranks.len());
    let mut attn_hc = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, input, _), block) in ranks.iter().zip(blocks.iter()) {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_attn_fn,
            &block.hc_attn_scale,
            &block.hc_attn_base,
        )?;
        attn_inputs.push(pre);
        attn_hc.push(state);
    }

    let mut attn_norms = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, attn_input)) in
        ranks.iter().zip(blocks.iter().zip(attn_inputs.iter()))
    {
        attn_norms.push(rms_norm_bf16_hidden(
            ctx,
            attn_input,
            &block.attn_norm,
            config.rms_norm_eps,
        )?);
    }

    let mut attention_group = ranks
        .iter()
        .zip(blocks.iter())
        .zip(attn_norms.iter())
        .zip(caches.iter_mut())
        .map(|((((ctx, _, comm, _, _), block), attn_norm), cache)| {
            (*ctx, &block.attn, *comm, attn_norm, cache)
        })
        .collect::<Vec<_>>();
    let attn_out = match config.compress_ratios[layer] {
        0 => {
            let mut out = Vec::with_capacity(attention_group.len());
            for ((ctx, attn, _comm, input, cache), rope) in
                attention_group.iter_mut().zip(ropes.iter())
            {
                out.push(attention_prefill_rank_local_bf16_hidden_with_cache(
                    ctx,
                    config,
                    layer,
                    input,
                    attn,
                    rope,
                    start_pos,
                    &mut cache.kv,
                )?);
            }
            let mut comms_and_hidden: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> =
                attention_group
                    .iter()
                    .zip(out.iter_mut())
                    .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
                    .collect();
            all_reduce_hidden_group_fp32(&mut comms_and_hidden)?;
            out
        }
        4 => attention_prefill_compressed_overlap_group_bf16_hidden_with_cache(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
        _ => attention_prefill_compressed_nonoverlap_group_bf16_hidden_with_cache(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )?,
    };

    let mut after_attn = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, input, _), attn_out), state)) in ranks
        .iter()
        .zip(attn_out.iter())
        .zip(attn_hc.iter())
        .enumerate()
    {
        after_attn.push(
            hc_post_bf16_hidden(ctx, attn_out, input, state)
                .with_context(|| format!("hc_post attention layer {layer} rank {rank}"))?,
        );
    }

    let mut ffn_inputs = Vec::with_capacity(ranks.len());
    let mut ffn_hc = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, input))) in ranks
        .iter()
        .zip(blocks.iter().zip(after_attn.iter()))
        .enumerate()
    {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_ffn_fn,
            &block.hc_ffn_scale,
            &block.hc_ffn_base,
        )
        .with_context(|| format!("hc_pre ffn layer {layer} rank {rank}"))?;
        ffn_inputs.push(pre);
        ffn_hc.push(state);
    }

    let mut ffn_norms = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, ffn_input))) in ranks
        .iter()
        .zip(blocks.iter().zip(ffn_inputs.iter()))
        .enumerate()
    {
        ffn_norms.push(
            rms_norm_bf16_hidden(ctx, ffn_input, &block.ffn_norm, config.rms_norm_eps)
                .with_context(|| format!("ffn rms_norm layer {layer} rank {rank}"))?,
        );
    }

    let moe_group = ranks
        .iter()
        .zip(ffn_norms.iter())
        .map(|((ctx, weights, comm, _, token_ids), ffn_norm)| {
            (*ctx, *weights, *comm, ffn_norm, *token_ids)
        })
        .collect::<Vec<_>>();
    let ffn_out = moe_group_bf16_hidden(&moe_group, config, layer)
        .with_context(|| format!("moe_group_bf16_hidden layer {layer}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for (rank, (((ctx, _, _, _, _), ffn_out), (input, state))) in ranks
        .iter()
        .zip(ffn_out.iter())
        .zip(after_attn.iter().zip(ffn_hc.iter()))
        .enumerate()
    {
        out.push(
            hc_post_bf16_hidden(ctx, ffn_out, input, state)
                .with_context(|| format!("hc_post ffn layer {layer} rank {rank}"))?,
        );
    }
    Ok(out)
}

pub fn block_decode_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &HcHiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
    ropes: &[&DeepSeekRopeCache],
    start_pos: usize,
    caches: &mut [LayerDecodeCache],
) -> Result<Vec<HcHiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "block decode group must contain at least one rank"
    );
    ensure!(
        ranks.len() == ropes.len(),
        "block decode ranks/ropes length mismatch: ranks={}, ropes={}",
        ranks.len(),
        ropes.len()
    );
    ensure!(
        ranks.len() == caches.len(),
        "block decode ranks/cache length mismatch: ranks={}, caches={}",
        ranks.len(),
        caches.len()
    );
    ensure!(
        layer < config.n_layers,
        "group block decode layer {layer} out of range"
    );
    let blocks = ranks
        .iter()
        .map(|(_, weights, _, _, _)| weights.block(layer))
        .collect::<Result<Vec<_>>>()?;

    let mut attn_inputs = Vec::with_capacity(ranks.len());
    let mut attn_hc = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, input, _), block)) in ranks.iter().zip(blocks.iter()).enumerate() {
        ensure!(
            input.seq_len == 1,
            "block decode expects HC seq_len=1, got {}",
            input.seq_len
        );
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_attn_fn,
            &block.hc_attn_scale,
            &block.hc_attn_base,
        )
        .with_context(|| format!("hc_pre attention layer {layer} rank {rank}"))?;
        attn_inputs.push(pre);
        attn_hc.push(state);
    }

    let mut attn_norms = Vec::with_capacity(ranks.len());
    for (rank, ((ctx, _, _, _, _), (block, attn_input))) in ranks
        .iter()
        .zip(blocks.iter().zip(attn_inputs.iter()))
        .enumerate()
    {
        attn_norms.push(
            rms_norm_bf16_hidden(ctx, attn_input, &block.attn_norm, config.rms_norm_eps)
                .with_context(|| format!("attention rms_norm layer {layer} rank {rank}"))?,
        );
    }

    let mut attention_group = ranks
        .iter()
        .zip(blocks.iter())
        .zip(attn_norms.iter())
        .zip(caches.iter_mut())
        .map(|((((ctx, _, comm, _, _), block), attn_norm), cache)| {
            (*ctx, &block.attn, *comm, attn_norm, cache)
        })
        .collect::<Vec<_>>();
    let attn_out = match config.compress_ratios[layer] {
        0 => {
            let mut out = Vec::with_capacity(attention_group.len());
            for (rank, ((ctx, attn, _comm, input, cache), rope)) in
                attention_group.iter_mut().zip(ropes.iter()).enumerate()
            {
                out.push(
                    attention_decode_rank_local_bf16_hidden(
                        ctx,
                        config,
                        layer,
                        input,
                        attn,
                        rope,
                        start_pos,
                        &mut cache.kv,
                    )
                    .with_context(|| {
                        format!("attention_decode_rank_local layer {layer} rank {rank}")
                    })?,
                );
            }
            let mut attn_reduce: Vec<(&RankGpuContext, &Comm, &mut Bf16HiddenStates)> =
                attention_group
                    .iter()
                    .zip(out.iter_mut())
                    .map(|((ctx, _, comm, _, _), hidden)| (*ctx, *comm, hidden))
                    .collect();
            all_reduce_hidden_group_fp32(&mut attn_reduce)
                .with_context(|| format!("attention all_reduce layer {layer}"))?;
            out
        }
        4 => attention_decode_compressed_overlap_group_bf16_hidden(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )
        .with_context(|| format!("attention_decode_compressed_overlap layer {layer}"))?,
        _ => attention_decode_compressed_nonoverlap_group_bf16_hidden(
            &mut attention_group,
            config,
            layer,
            ropes,
            start_pos,
        )
        .with_context(|| format!("attention_decode_compressed_nonoverlap layer {layer}"))?,
    };

    let mut after_attn = Vec::with_capacity(ranks.len());
    for (((ctx, _, _, input, _), attn_out), state) in
        ranks.iter().zip(attn_out.iter()).zip(attn_hc.iter())
    {
        after_attn.push(hc_post_bf16_hidden(ctx, attn_out, input, state)?);
    }

    let mut ffn_inputs = Vec::with_capacity(ranks.len());
    let mut ffn_hc = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, input)) in
        ranks.iter().zip(blocks.iter().zip(after_attn.iter()))
    {
        let (pre, state) = hc_pre_bf16_hidden(
            ctx,
            config,
            input,
            &block.hc_ffn_fn,
            &block.hc_ffn_scale,
            &block.hc_ffn_base,
        )?;
        ffn_inputs.push(pre);
        ffn_hc.push(state);
    }

    let mut ffn_norms = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (block, ffn_input)) in
        ranks.iter().zip(blocks.iter().zip(ffn_inputs.iter()))
    {
        ffn_norms.push(rms_norm_bf16_hidden(
            ctx,
            ffn_input,
            &block.ffn_norm,
            config.rms_norm_eps,
        )?);
    }

    let moe_group = ranks
        .iter()
        .zip(ffn_norms.iter())
        .map(|((ctx, weights, comm, _, token_ids), ffn_norm)| {
            (*ctx, *weights, *comm, ffn_norm, *token_ids)
        })
        .collect::<Vec<_>>();
    let ffn_out = moe_group_bf16_hidden(&moe_group, config, layer)?;

    let mut out = Vec::with_capacity(ranks.len());
    for (((ctx, _, _, _, _), ffn_out), (input, state)) in ranks
        .iter()
        .zip(ffn_out.iter())
        .zip(after_attn.iter().zip(ffn_hc.iter()))
    {
        out.push(hc_post_bf16_hidden(ctx, ffn_out, input, state)?);
    }
    Ok(out)
}

pub fn prefill_logits_group_bf16_hidden(
    ranks: &[(&RankGpuContext, &RankWeightView<'_>, &Comm, &CudaSlice<u32>)],
    config: &Config,
    seq_len: usize,
) -> Result<Vec<F32Logits>> {
    ensure!(
        !ranks.is_empty(),
        "full prefill group must contain at least one rank"
    );
    ensure!(seq_len > 0, "full prefill seq_len must be positive");

    let hidden = embedding_vocab_parallel_group(ranks, config, seq_len)?;
    let mut hcs = ranks
        .iter()
        .zip(hidden.iter())
        .map(|((ctx, _, _, _), hidden)| hc_expand_bf16_hidden(ctx, hidden, config.hc_mult))
        .collect::<Result<Vec<_>>>()?;

    for layer in 0..config.n_layers {
        let ropes = ranks
            .iter()
            .map(|(ctx, _, _, _)| precompute_rope_cache(ctx, config, layer, seq_len))
            .collect::<Result<Vec<_>>>()?;
        let rope_refs = ropes.iter().collect::<Vec<_>>();
        let block_inputs = ranks
            .iter()
            .zip(hcs.iter())
            .map(|((ctx, weights, comm, token_ids), hc)| (*ctx, *weights, *comm, hc, *token_ids))
            .collect::<Vec<_>>();
        hcs = block_prefill_group_bf16_hidden(&block_inputs, config, layer, &rope_refs, 0)?;
    }

    let logits_inputs = ranks
        .iter()
        .zip(hcs.iter())
        .map(|((ctx, weights, comm, _), hc)| (*ctx, *weights, *comm, hc))
        .collect::<Vec<_>>();
    final_logits_group_bf16_hidden(&logits_inputs, config)
}

pub fn prefill_logits_and_decode_cache_group_bf16_hidden(
    ranks: &[(&RankGpuContext, &RankWeightView<'_>, &Comm, &CudaSlice<u32>)],
    config: &Config,
    seq_len: usize,
    caches: &mut [Vec<LayerDecodeCache>],
) -> Result<Vec<F32Logits>> {
    ensure!(
        !ranks.is_empty(),
        "full prefill cache group must contain at least one rank"
    );
    ensure!(seq_len > 0, "full prefill cache seq_len must be positive");
    ensure!(
        caches.len() == config.n_layers,
        "prefill cache layer count mismatch: have {}, need {}",
        caches.len(),
        config.n_layers
    );
    for (layer, rank_caches) in caches.iter().enumerate() {
        ensure!(
            rank_caches.len() == ranks.len(),
            "prefill cache rank count mismatch at layer {layer}: have {}, need {}",
            rank_caches.len(),
            ranks.len()
        );
    }

    let hidden = embedding_vocab_parallel_group(ranks, config, seq_len)?;
    let mut hcs = ranks
        .iter()
        .zip(hidden.iter())
        .map(|((ctx, _, _, _), hidden)| hc_expand_bf16_hidden(ctx, hidden, config.hc_mult))
        .collect::<Result<Vec<_>>>()?;

    for (layer, layer_caches) in caches.iter_mut().enumerate().take(config.n_layers) {
        let ropes = ranks
            .iter()
            .map(|(ctx, _, _, _)| precompute_rope_cache(ctx, config, layer, seq_len))
            .collect::<Result<Vec<_>>>()?;
        let rope_refs = ropes.iter().collect::<Vec<_>>();
        let block_inputs = ranks
            .iter()
            .zip(hcs.iter())
            .map(|((ctx, weights, comm, token_ids), hc)| (*ctx, *weights, *comm, hc, *token_ids))
            .collect::<Vec<_>>();
        hcs = block_prefill_group_bf16_hidden_with_decode_cache(
            &block_inputs,
            config,
            layer,
            &rope_refs,
            0,
            layer_caches,
        )?;
    }

    let logits_inputs = ranks
        .iter()
        .zip(hcs.iter())
        .map(|((ctx, weights, comm, _), hc)| (*ctx, *weights, *comm, hc))
        .collect::<Vec<_>>();
    final_logits_group_bf16_hidden(&logits_inputs, config)
}

pub fn hash_route_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
    token_ids: &CudaSlice<u32>,
) -> Result<RoutedExperts> {
    ctx.set_current()?;
    ensure!(
        ffn.gate_weight.tensor.dtype == safetensors::Dtype::BF16,
        "gate weight {} must be BF16, got {:?}",
        ffn.gate_weight.name,
        ffn.gate_weight.tensor.dtype
    );
    ensure!(
        ffn.gate_weight.tensor.shape == [config.n_routed_experts, input.hidden_dim],
        "gate weight {} shape mismatch: expected {:?}, got {:?}",
        ffn.gate_weight.name,
        [config.n_routed_experts, input.hidden_dim],
        ffn.gate_weight.tensor.shape
    );
    let tid2eid = ffn
        .gate_tid2eid
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("hash routing requires gate.tid2eid"))?;
    ensure!(
        tid2eid.tensor.dtype == safetensors::Dtype::I64,
        "gate tid2eid {} must be I64, got {:?}",
        tid2eid.name,
        tid2eid.tensor.dtype
    );
    ensure!(
        tid2eid.tensor.shape == [config.vocab_size, config.n_activated_experts],
        "gate tid2eid {} shape mismatch: expected {:?}, got {:?}",
        tid2eid.name,
        [config.vocab_size, config.n_activated_experts],
        tid2eid.tensor.shape
    );

    let mut weights = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    let mut indices = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let (tid2eid_ptr, _tid2eid_guard) = tid2eid.tensor.data.device_ptr(&ctx.stream);
        let (token_ptr, _token_guard) = token_ids.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_hash_gate_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                tid2eid_ptr as *const i64,
                token_ptr as *const u32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(RoutedExperts {
        weights,
        indices,
        topk: config.n_activated_experts,
        seq_len: input.seq_len,
    })
}

pub fn score_route_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    input: &Bf16HiddenStates,
    ffn: &FfnWeights<'_>,
) -> Result<RoutedExperts> {
    ctx.set_current()?;
    ensure!(
        ffn.gate_weight.tensor.dtype == safetensors::Dtype::BF16,
        "gate weight {} must be BF16, got {:?}",
        ffn.gate_weight.name,
        ffn.gate_weight.tensor.dtype
    );
    ensure!(
        ffn.gate_weight.tensor.shape == [config.n_routed_experts, input.hidden_dim],
        "gate weight {} shape mismatch: expected {:?}, got {:?}",
        ffn.gate_weight.name,
        [config.n_routed_experts, input.hidden_dim],
        ffn.gate_weight.tensor.shape
    );
    let bias = ffn
        .gate_bias
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("score routing requires gate.bias"))?;
    ensure!(
        bias.tensor.dtype == safetensors::Dtype::F32,
        "gate bias {} must be F32, got {:?}",
        bias.name,
        bias.tensor.dtype
    );
    ensure!(
        bias.tensor.shape == [config.n_routed_experts],
        "gate bias {} shape mismatch: expected {:?}, got {:?}",
        bias.name,
        [config.n_routed_experts],
        bias.tensor.shape
    );

    let mut weights = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    let mut indices = ctx
        .stream
        .alloc_zeros(input.seq_len * config.n_activated_experts)?;
    {
        let (x_ptr, _x_guard) = input.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gate_guard) = ffn.gate_weight.tensor.data.device_ptr(&ctx.stream);
        let (bias_ptr, _bias_guard) = bias.tensor.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
        let (indices_ptr, _indices_guard) = indices.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_score_gate_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                bias_ptr as *const f32,
                weights_ptr as *mut f32,
                indices_ptr as *mut i32,
                input.seq_len as i32,
                input.hidden_dim as i32,
                config.n_routed_experts as i32,
                config.n_activated_experts as i32,
                config.routed_scaling_factor,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(RoutedExperts {
        weights,
        indices,
        topk: config.n_activated_experts,
        seq_len: input.seq_len,
    })
}

pub fn accumulate_weighted_expert_output(
    ctx: &RankGpuContext,
    expert_out: &Bf16HiddenStates,
    routed: &RoutedExperts,
    global_expert: usize,
    accum: &mut Bf16HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        expert_out.hidden_dim == accum.hidden_dim,
        "expert output dim mismatch: expert={}, accum={}",
        expert_out.hidden_dim,
        accum.hidden_dim
    );
    ensure!(
        expert_out.seq_len == accum.seq_len,
        "expert output seq len mismatch: expert={}, accum={}",
        expert_out.seq_len,
        accum.seq_len
    );
    ensure!(
        routed.seq_len == expert_out.seq_len,
        "route seq len mismatch: route={}, expert={}",
        routed.seq_len,
        expert_out.seq_len
    );

    {
        let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
        let (weights_ptr, _weights_guard) = routed.weights.device_ptr(&ctx.stream);
        let (indices_ptr, _indices_guard) = routed.indices.device_ptr(&ctx.stream);
        let (accum_ptr, _accum_guard) = accum.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_weighted_expert_accum_cuda(
                expert_ptr as *const ffi::Half,
                weights_ptr as *const f32,
                indices_ptr as *const i32,
                accum_ptr as *mut ffi::Half,
                expert_out.seq_len as i32,
                expert_out.hidden_dim as i32,
                routed.topk as i32,
                global_expert as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn add_f32_bf16_to_bf16_hidden(
    ctx: &RankGpuContext,
    a: &F32HiddenStates,
    b: &Bf16HiddenStates,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    ensure!(
        a.hidden_dim == b.hidden_dim,
        "add f32/bf16 hidden dim mismatch: a={}, b={}",
        a.hidden_dim,
        b.hidden_dim
    );
    ensure!(
        a.seq_len == b.seq_len,
        "add f32/bf16 seq len mismatch: a={}, b={}",
        a.seq_len,
        b.seq_len
    );
    let mut out = Bf16HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    {
        let (a_ptr, _a_guard) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_add_f32_bf16_to_bf16_cuda(
                a_ptr as *const f32,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                (a.hidden_dim * a.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(out)
}

pub fn accumulate_expert_output_f32(
    ctx: &RankGpuContext,
    expert_out: &Bf16HiddenStates,
    accum: &mut F32HiddenStates,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        expert_out.hidden_dim == accum.hidden_dim,
        "expert output dim mismatch: expert={}, accum={}",
        expert_out.hidden_dim,
        accum.hidden_dim
    );
    ensure!(
        expert_out.seq_len == accum.seq_len,
        "expert output seq len mismatch: expert={}, accum={}",
        expert_out.seq_len,
        accum.seq_len
    );
    {
        let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
        let (accum_ptr, _accum_guard) = accum.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_expert_accum_f32_cuda(
                expert_ptr as *const ffi::Half,
                accum_ptr as *mut f32,
                (expert_out.hidden_dim * expert_out.seq_len) as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}

pub fn routed_local_experts_forward_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    routed: &RoutedExperts,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let mut out = Bf16HiddenStates::zeros(ctx, input.hidden_dim, input.seq_len)?;
    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let route_indices = ctx.stream.clone_dtoh(&routed.indices)?;
    ctx.sync()?;
    let mut active_local = vec![false; local_experts];
    for expert in route_indices {
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        if (global_start..global_start + local_experts).contains(&expert) {
            active_local[expert - global_start] = true;
        }
    }
    for local_expert in 0..local_experts {
        if !active_local[local_expert] {
            continue;
        }
        let expert = weights.local_expert(layer, local_expert)?;
        let expert_out =
            local_expert_forward_bf16_hidden(ctx, input, &expert, config.swiglu_limit)?;
        accumulate_weighted_expert_output(
            ctx,
            &expert_out,
            routed,
            global_start + local_expert,
            &mut out,
        )?;
    }
    Ok(out)
}

pub fn routed_local_experts_forward_f32_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    routed: &RoutedExperts,
) -> Result<F32HiddenStates> {
    ctx.set_current()?;
    let mut out = F32HiddenStates {
        data: ctx.stream.alloc_zeros(input.hidden_dim * input.seq_len)?,
        hidden_dim: input.hidden_dim,
        seq_len: input.seq_len,
    };
    let local_experts = config.n_routed_experts / weights.world_size();
    let global_start = weights.rank() * local_experts;
    let route_indices = ctx.stream.clone_dtoh(&routed.indices)?;
    ctx.sync()?;
    let mut active_local = vec![false; local_experts];
    for expert in route_indices {
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        if (global_start..global_start + local_experts).contains(&expert) {
            active_local[expert - global_start] = true;
        }
    }
    for local_expert in 0..local_experts {
        if !active_local[local_expert] {
            continue;
        }
        let global_expert = global_start + local_expert;
        let expert = weights.local_expert(layer, local_expert)?;
        let expert_out = local_expert_forward_weighted_bf16_hidden(
            ctx,
            input,
            &expert,
            routed,
            global_expert,
            config.swiglu_limit,
        )?;
        accumulate_expert_output_f32(ctx, &expert_out, &mut out)?;
    }
    Ok(out)
}

pub fn hash_routed_moe_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?;
    let routed_out =
        routed_local_experts_forward_bf16_hidden(ctx, config, weights, layer, input, &routed)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_bf16_hidden(ctx, &routed_out, &shared)
}

pub fn moe_rank_local_bf16_hidden(
    ctx: &RankGpuContext,
    config: &Config,
    weights: &RankWeightView<'_>,
    layer: usize,
    input: &Bf16HiddenStates,
    token_ids: &CudaSlice<u32>,
) -> Result<Bf16HiddenStates> {
    ctx.set_current()?;
    let ffn = weights.ffn(layer)?;
    let routed = if layer < config.n_hash_layers {
        hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
    } else {
        score_route_bf16_hidden(ctx, config, input, &ffn)?
    };
    let routed_out =
        routed_local_experts_forward_bf16_hidden(ctx, config, weights, layer, input, &routed)?;
    let shared = shared_expert_forward_bf16_hidden(ctx, input, &ffn, config.swiglu_limit)?;
    add_bf16_hidden(ctx, &routed_out, &shared)
}

pub fn hash_routed_moe_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &Bf16HiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "MoE group must contain at least one rank"
    );

    let mut routed_out = Vec::with_capacity(ranks.len());
    let mut shared_out = Vec::with_capacity(ranks.len());
    for (ctx, weights, _comm, input, token_ids) in ranks {
        let ffn = weights.ffn(layer)?;
        let routed = hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?;
        routed_out.push(routed_local_experts_forward_f32_hidden(
            ctx, config, weights, layer, input, &routed,
        )?);
        shared_out.push(shared_expert_forward_bf16_hidden(
            ctx,
            input,
            &ffn,
            config.swiglu_limit,
        )?);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _, _), hidden) in ranks.iter().zip(routed_out.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!(
                "NCCL MoE routed all-reduce failed: {err:?}"
            ));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (routed, shared)) in
        ranks.iter().zip(routed_out.iter().zip(shared_out.iter()))
    {
        out.push(add_f32_bf16_to_bf16_hidden(ctx, routed, shared)?);
    }
    Ok(out)
}

pub fn moe_group_bf16_hidden(
    ranks: &[(
        &RankGpuContext,
        &RankWeightView<'_>,
        &Comm,
        &Bf16HiddenStates,
        &CudaSlice<u32>,
    )],
    config: &Config,
    layer: usize,
) -> Result<Vec<Bf16HiddenStates>> {
    ensure!(
        !ranks.is_empty(),
        "MoE group must contain at least one rank"
    );

    let mut routed_out = Vec::with_capacity(ranks.len());
    let mut shared_out = Vec::with_capacity(ranks.len());
    for (ctx, weights, _comm, input, token_ids) in ranks {
        let ffn = weights.ffn(layer)?;
        let routed = if layer < config.n_hash_layers {
            hash_route_bf16_hidden(ctx, config, input, &ffn, token_ids)?
        } else {
            score_route_bf16_hidden(ctx, config, input, &ffn)?
        };
        routed_out.push(routed_local_experts_forward_f32_hidden(
            ctx, config, weights, layer, input, &routed,
        )?);
        shared_out.push(shared_expert_forward_bf16_hidden(
            ctx,
            input,
            &ffn,
            config.swiglu_limit,
        )?);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, _, comm, _, _), hidden) in ranks.iter().zip(routed_out.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!(
                "NCCL MoE routed all-reduce failed: {err:?}"
            ));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    let mut out = Vec::with_capacity(ranks.len());
    for ((ctx, _, _, _, _), (routed, shared)) in
        ranks.iter().zip(routed_out.iter().zip(shared_out.iter()))
    {
        out.push(add_f32_bf16_to_bf16_hidden(ctx, routed, shared)?);
    }
    Ok(out)
}

pub fn all_reduce_hidden_in_place(hidden: &mut Bf16HiddenStates, comm: &Comm) -> Result<()> {
    comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
        .map_err(|err| anyhow::anyhow!("NCCL all-reduce failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_hidden_group(
    comms_and_hidden: &mut [(&Comm, &mut Bf16HiddenStates)],
) -> Result<()> {
    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for (comm, hidden) in comms_and_hidden {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_hidden_group_bf16(
    contexts_comms_and_hidden: &mut [(&RankGpuContext, &Comm, &mut Bf16HiddenStates)],
) -> Result<()> {
    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for (_, comm, hidden) in contexts_comms_and_hidden {
        if let Err(err) = comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL BF16 all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;
    Ok(())
}

pub fn all_reduce_hidden_group_fp32(
    contexts_comms_and_hidden: &mut [(&RankGpuContext, &Comm, &mut Bf16HiddenStates)],
) -> Result<()> {
    let mut temps = Vec::<CudaSlice<f32>>::with_capacity(contexts_comms_and_hidden.len());
    for (ctx, _, hidden) in contexts_comms_and_hidden.iter_mut() {
        let len = hidden.hidden_dim * hidden.seq_len;
        ctx.set_current()?;
        let mut temp = ctx.stream.alloc_zeros::<f32>(len)?;
        {
            let (input_ptr, _input_guard) = hidden.data.device_ptr(&ctx.stream);
            let (temp_ptr, _temp_guard) = temp.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                ffi::deepseek_bf16_to_f32_cuda(
                    input_ptr as *const ffi::Half,
                    temp_ptr as *mut f32,
                    len as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result()?;
        }
        temps.push(temp);
    }

    group_start().map_err(|err| anyhow::anyhow!("NCCL group_start failed: {err:?}"))?;
    for ((_, comm, _), temp) in contexts_comms_and_hidden.iter_mut().zip(temps.iter_mut()) {
        if let Err(err) = comm.all_reduce_in_place(temp, &ReduceOp::Sum) {
            let _ = group_end();
            return Err(anyhow::anyhow!("NCCL FP32 all-reduce failed: {err:?}"));
        }
    }
    group_end().map_err(|err| anyhow::anyhow!("NCCL group_end failed: {err:?}"))?;

    for ((ctx, _, hidden), temp) in contexts_comms_and_hidden.iter_mut().zip(temps.iter()) {
        let len = hidden.hidden_dim * hidden.seq_len;
        ctx.set_current()?;
        let (temp_ptr, _temp_guard) = temp.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = hidden.data.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_f32_to_bf16_cuda(
                temp_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(())
}
