use super::*;

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
