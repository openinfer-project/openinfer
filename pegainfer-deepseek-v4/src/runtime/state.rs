use super::*;

pub struct Bf16HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl Bf16HiddenStates {
    pub(crate) fn seq_capacity(&self) -> usize {
        if self.hidden_dim == 0 {
            0
        } else {
            self.data.len() / self.hidden_dim
        }
    }
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

impl HcHiddenStates {
    pub(crate) fn seq_capacity(&self) -> usize {
        if self.hidden_dim == 0 || self.hc == 0 {
            0
        } else {
            self.data.len() / (self.hidden_dim * self.hc)
        }
    }
}

pub struct HcPreState {
    pub post: CudaSlice<f32>,
    pub comb: CudaSlice<f32>,
    pub seq_len: usize,
    pub hc: usize,
}

pub(crate) struct HcPreStateView<'a> {
    pub post: &'a CudaSlice<f32>,
    pub comb: &'a CudaSlice<f32>,
    pub seq_len: usize,
    pub hc: usize,
}

pub(crate) struct HcPreNormScratch {
    pub mixes: CudaSlice<f32>,
    pub post: CudaSlice<f32>,
    pub comb: CudaSlice<f32>,
    pub out: Bf16HiddenStates,
    pub seq_capacity: usize,
    pub hidden_dim: usize,
    pub hc: usize,
}

pub(crate) struct DecodeEntryScratch {
    pub(crate) embedding: Bf16HiddenStates,
    pub(crate) hc_expand: HcHiddenStates,
}

pub(crate) struct DecodeBatchMeta<'a> {
    pub(crate) batch: usize,
    pub(crate) compressed_slots: usize,
    pub(crate) start_pos: &'a CudaSlice<i32>,
    pub(crate) src_rows: &'a CudaSlice<i32>,
    pub(crate) window_dst_rows: &'a CudaSlice<i32>,
    pub(crate) window_base: &'a CudaSlice<i32>,
    pub(crate) compressed_base: &'a CudaSlice<i32>,
    pub(crate) compressed_len: &'a CudaSlice<i32>,
    pub(crate) start_pos_host: &'a [usize],
    pub(crate) slot_ids_host: &'a [usize],
}

pub(crate) fn decode_cache_compressed_row(
    sliding_window: usize,
    compressed_slots: usize,
    slot_id: usize,
    compressed_row: usize,
) -> usize {
    slot_id * (sliding_window + compressed_slots) + sliding_window + compressed_row
}

pub(crate) struct HcPostScratch {
    pub(crate) attention_reduce_temp: CudaSlice<f32>,
    pub(crate) attention_out: HcHiddenStates,
    pub(crate) layer_outputs: Vec<HcHiddenStates>,
}

pub(crate) struct FinalLogitsScratch {
    pub(crate) hc_mixes: CudaSlice<f32>,
    pub(crate) hc_pre: CudaSlice<f32>,
    pub(crate) hc_out: Bf16HiddenStates,
    pub(crate) normed: Bf16HiddenStates,
    pub(crate) local_logits: F32Logits,
    pub(crate) gathered_logits: F32Logits,
}

/// Per-decode-step MoE-side context handed to `block_decode_rank_lane_*`.
///
/// Holds the always-present NCCL all-reduce / AG-RS lane plus, when the
/// `pplx-ep` feature is on, an optional pplx-garden EP lane. The block
/// decode helper routes the routed-expert step to whichever lane is
/// configured; both lanes are mutually exclusive within a single layer.
pub(crate) struct MoeRunContext<'a> {
    pub(crate) moe_comm: &'a cudarc::nccl::safe::Comm,
    pub(crate) ag_rs_scratch: &'a mut MoeAgRsScratch,
    #[cfg(feature = "pplx-ep")]
    pub(crate) pplx: Option<MoePplxRunContext<'a>>,
}

/// Per-decode-step pplx-lane bundle. Borrows the rank worker's persistent
/// `EpBackend` and pplx scratch.
#[cfg(feature = "pplx-ep")]
pub(crate) struct MoePplxRunContext<'a> {
    pub(crate) ep: &'a mut pegainfer_comm::EpBackend,
    pub(crate) scratch: &'a mut MoePplxScratch,
}

/// Scratch + MR-registered staging memory for the pplx-garden EP all-to-all
/// decode path. Field set mirrors only what the pplx flow actually consumes
/// — no global hidden, no partial/local routed F32 buffers (the combine
/// kernel reduces in-place into the BF16 output).
#[cfg(feature = "pplx-ep")]
pub(crate) struct MoePplxScratch {
    /// Local route output: top-k weights `[seq_len, topk]`.
    pub(crate) route_weights: cudarc::driver::CudaSlice<f32>,
    /// Local route output: top-k global expert indices `[seq_len, topk]`.
    pub(crate) route_indices: cudarc::driver::CudaSlice<i32>,
    /// Receiver-side expert-packed activation. Sized for `max_recv_tokens`
    /// rows; `dispatch_recv` writes here.
    pub(crate) expanded_input: Bf16HiddenStates,
    /// Grouped W1 output scratch (intermediate dim).
    pub(crate) expert_gate: Bf16HiddenStates,
    /// Grouped W3 output scratch (intermediate dim).
    pub(crate) expert_up: Bf16HiddenStates,
    /// Grouped W2 + SwiGLU output (hidden dim). Fed into `combine_send`.
    pub(crate) expert_out: Bf16HiddenStates,
    /// FP4 activation workspace shared between grouped GEMMs.
    pub(crate) fp4_act_workspace: cudarc::driver::CudaSlice<u8>,
    /// FP4 activation-scale workspace shared between grouped GEMMs.
    pub(crate) fp4_act_scale_workspace: cudarc::driver::CudaSlice<u8>,
    /// Exclusive prefix sum of received rows per local expert
    /// (`local_experts + 1` entries). Built on device from `dispatch_recv`'s
    /// per-expert count output.
    pub(crate) expert_indptr: cudarc::driver::CudaSlice<i32>,
    /// Per-local-expert token counts written by `dispatch_recv`.
    pub(crate) recv_tokens_per_expert: cudarc::driver::CudaSlice<i32>,
    /// Per-expert alignment used by pplx's padded receive layout.
    pub(crate) expert_padding: usize,
    /// Final BF16 output `[seq_len, hidden_dim]`. Shared-expert result is
    /// staged here first; `combine_recv` then runs with `accumulate=true`
    /// to fold in the routed contribution.
    pub(crate) out: Bf16HiddenStates,
}

pub(crate) struct MoeAgRsScratch {
    pub(crate) global_hidden: Bf16HiddenStates,
    pub(crate) global_token_ids: CudaSlice<u32>,
    pub(crate) route_weights: CudaSlice<f32>,
    pub(crate) route_indices: CudaSlice<i32>,
    pub(crate) pos_to_token: CudaSlice<i32>,
    pub(crate) pos_to_token_topk: CudaSlice<i32>,
    pub(crate) token_topk_to_pos: CudaSlice<i32>,
    pub(crate) expert_indptr: CudaSlice<i32>,
    pub(crate) expert_cursor: CudaSlice<i32>,
    pub(crate) local_count: CudaSlice<i32>,
    pub(crate) expanded_input: Bf16HiddenStates,
    pub(crate) expert_gate: Bf16HiddenStates,
    pub(crate) expert_up: Bf16HiddenStates,
    pub(crate) expert_out: Bf16HiddenStates,
    pub(crate) fp4_act_workspace: CudaSlice<u8>,
    pub(crate) fp4_act_scale_workspace: CudaSlice<u8>,
    pub(crate) partial_routed: F32HiddenStates,
    pub(crate) local_routed: F32HiddenStates,
    pub(crate) out: Bf16HiddenStates,
}

pub(crate) struct SharedExpertScratch {
    pub(crate) gate: Bf16HiddenStates,
    pub(crate) up: Bf16HiddenStates,
    pub(crate) out: Bf16HiddenStates,
    pub(crate) fp8_act_workspace: CudaSlice<u8>,
    pub(crate) fp8_act_scale_workspace: CudaSlice<u8>,
    pub(crate) seq_capacity: usize,
}

pub(crate) struct AttentionOutputScratch {
    pub(crate) attn_out: Bf16HiddenStates,
    pub(crate) low_rank: Bf16HiddenStates,
    pub(crate) out: Bf16HiddenStates,
    pub(crate) local_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) seq_capacity: usize,
}

pub(crate) struct AttentionIndexScratch {
    pub(crate) window_idxs: CudaSlice<i32>,
    pub(crate) compress_idxs: CudaSlice<i32>,
    pub(crate) topk_idxs: CudaSlice<i32>,
}

pub(crate) struct AttentionAuxScratch {
    pub(crate) compressor_weighted: CudaSlice<f32>,
    pub(crate) compressor_out: Bf16HiddenStates,
    pub(crate) indexer_q: Bf16HiddenStates,
    pub(crate) indexer_weights: Bf16HiddenStates,
    pub(crate) indexer_scores: CudaSlice<f32>,
    pub(crate) max_head_dim: usize,
    pub(crate) local_index_heads: usize,
    pub(crate) max_compressed_len: usize,
}

pub struct F32Logits {
    pub data: CudaSlice<f32>,
    pub vocab_size: usize,
}

pub(crate) struct F32BatchLogits {
    pub(crate) data: CudaSlice<f32>,
    pub(crate) vocab_size: usize,
    pub(crate) seq_len: usize,
}

pub struct RoutedExperts {
    pub weights: CudaSlice<f32>,
    pub indices: CudaSlice<i32>,
    pub topk: usize,
    pub seq_len: usize,
}

pub(crate) struct RoutedExpertsView<'a> {
    pub(crate) weights: &'a CudaSlice<f32>,
    pub(crate) indices: &'a CudaSlice<i32>,
    pub(crate) topk: usize,
    pub(crate) seq_len: usize,
}

pub struct MoeFusedRoutePlan {
    pub routed: RoutedExperts,
    pub pos_to_token: CudaSlice<i32>,
    pub pos_to_token_topk: CudaSlice<i32>,
    pub token_topk_to_pos: CudaSlice<i32>,
    pub expert_indptr: CudaSlice<i32>,
    pub expert_cursor: CudaSlice<i32>,
    pub local_count: CudaSlice<i32>,
    pub local_experts: usize,
    pub global_start: usize,
    pub num_expanded: usize,
}

pub(crate) struct MoeFusedRoutePlanView<'a> {
    pub(crate) routed: RoutedExpertsView<'a>,
    pub(crate) pos_to_token: &'a CudaSlice<i32>,
    pub(crate) token_topk_to_pos: &'a CudaSlice<i32>,
    pub(crate) expert_indptr: &'a CudaSlice<i32>,
    pub(crate) local_experts: usize,
    pub(crate) num_expanded: usize,
}

pub struct MoeGroupedLinearPtrs {
    pub weight_ptrs: CudaSlice<u64>,
    pub scale_ptrs: CudaSlice<u64>,
    pub in_dim: usize,
    pub out_dim: usize,
}

pub struct MoeLayerGroupedPtrs {
    pub w1: MoeGroupedLinearPtrs,
    pub w2: MoeGroupedLinearPtrs,
    pub w3: MoeGroupedLinearPtrs,
}

pub struct MoeGroupedPtrCache {
    pub layers: Vec<MoeLayerGroupedPtrs>,
    pub local_experts: usize,
}

pub struct F32HiddenStates {
    pub data: CudaSlice<f32>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl F32HiddenStates {
    pub(crate) fn seq_capacity(&self) -> usize {
        if self.hidden_dim == 0 {
            0
        } else {
            self.data.len() / self.hidden_dim
        }
    }
}

pub struct AttentionProjections {
    pub qr: Bf16HiddenStates,
    pub q: Bf16HiddenStates,
    pub kv: Bf16HiddenStates,
    pub local_heads: usize,
    pub head_dim: usize,
}

pub(crate) struct AttentionProjectionsView<'a> {
    pub(crate) qr: &'a Bf16HiddenStates,
    pub(crate) q: &'a mut Bf16HiddenStates,
    pub(crate) kv: &'a mut Bf16HiddenStates,
    pub(crate) local_heads: usize,
    pub(crate) head_dim: usize,
}

pub(crate) struct AttentionProjectionScratch {
    pub(crate) qr_raw: Bf16HiddenStates,
    pub(crate) qr: Bf16HiddenStates,
    pub(crate) q_raw: Bf16HiddenStates,
    pub(crate) q: Bf16HiddenStates,
    pub(crate) kv_raw: Bf16HiddenStates,
    pub(crate) kv: Bf16HiddenStates,
    pub(crate) local_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) seq_capacity: usize,
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

    pub fn uninit(ctx: &RankGpuContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        ctx.set_current()?;
        let data = unsafe { ctx.stream.alloc(hidden_dim * seq_len)? };
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

impl HcPreNormScratch {
    pub(crate) fn new(ctx: &RankGpuContext, config: &Config, seq_capacity: usize) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "HC pre-norm scratch capacity must be positive"
        );
        let mix_hc = (2 + config.hc_mult) * config.hc_mult;
        let mixes = unsafe { ctx.stream.alloc(seq_capacity * mix_hc)? };
        let post = unsafe { ctx.stream.alloc(seq_capacity * config.hc_mult)? };
        let comb = unsafe {
            ctx.stream
                .alloc(seq_capacity * config.hc_mult * config.hc_mult)?
        };
        let out = Bf16HiddenStates::uninit(ctx, config.dim, seq_capacity)?;
        Ok(Self {
            mixes,
            post,
            comb,
            out,
            seq_capacity,
            hidden_dim: config.dim,
            hc: config.hc_mult,
        })
    }
}

impl DecodeEntryScratch {
    pub(crate) fn new(ctx: &RankGpuContext, config: &Config, seq_capacity: usize) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "decode entry scratch capacity must be positive"
        );
        let embedding = Bf16HiddenStates::uninit(ctx, config.dim, seq_capacity)?;
        let hc_expand = HcHiddenStates::uninit(ctx, config.dim, seq_capacity, config.hc_mult)?;
        Ok(Self {
            embedding,
            hc_expand,
        })
    }
}

impl HcPostScratch {
    pub(crate) fn new(ctx: &RankGpuContext, config: &Config, seq_capacity: usize) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "HC post scratch capacity must be positive"
        );
        let attention_reduce_temp = unsafe { ctx.stream.alloc(config.dim * seq_capacity)? };
        let attention_out = HcHiddenStates::uninit(ctx, config.dim, seq_capacity, config.hc_mult)?;
        let mut layer_outputs = Vec::with_capacity(2);
        for _ in 0..2 {
            layer_outputs.push(HcHiddenStates::uninit(
                ctx,
                config.dim,
                seq_capacity,
                config.hc_mult,
            )?);
        }
        Ok(Self {
            attention_reduce_temp,
            attention_out,
            layer_outputs,
        })
    }
}

impl FinalLogitsScratch {
    pub(crate) fn new(
        ctx: &RankGpuContext,
        config: &Config,
        world_size: usize,
        seq_capacity: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "final logits scratch capacity must be positive"
        );
        ensure!(
            world_size > 0,
            "final logits scratch world size must be positive"
        );
        ensure!(
            config.vocab_size.is_multiple_of(world_size),
            "final logits scratch vocab_size={} not divisible by world_size={world_size}",
            config.vocab_size
        );
        let local_vocab_size = config.vocab_size / world_size;
        let hc_mixes = unsafe { ctx.stream.alloc(seq_capacity * config.hc_mult)? };
        let hc_pre = unsafe { ctx.stream.alloc(seq_capacity * config.hc_mult)? };
        let hc_out = Bf16HiddenStates::uninit(ctx, config.dim, seq_capacity)?;
        let normed = Bf16HiddenStates::uninit(ctx, config.dim, seq_capacity)?;
        let local_logits = F32Logits {
            data: unsafe { ctx.stream.alloc(local_vocab_size)? },
            vocab_size: local_vocab_size,
        };
        let gathered_logits = F32Logits {
            data: unsafe { ctx.stream.alloc(config.vocab_size)? },
            vocab_size: config.vocab_size,
        };
        Ok(Self {
            hc_mixes,
            hc_pre,
            hc_out,
            normed,
            local_logits,
            gathered_logits,
        })
    }
}

impl MoeAgRsScratch {
    pub(crate) fn new(
        ctx: &RankGpuContext,
        config: &Config,
        world_size: usize,
        local_seq_capacity: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            world_size > 0,
            "MoE AG/RS scratch world size must be positive"
        );
        ensure!(
            local_seq_capacity > 0,
            "MoE AG/RS scratch local capacity must be positive"
        );
        ensure!(
            config.n_routed_experts.is_multiple_of(world_size),
            "MoE AG/RS scratch n_routed_experts={} not divisible by world_size={world_size}",
            config.n_routed_experts
        );
        let global_seq_capacity = local_seq_capacity * world_size;
        let global_hidden = Bf16HiddenStates::uninit(ctx, config.dim, global_seq_capacity)?;
        let global_token_ids = unsafe { ctx.stream.alloc(global_seq_capacity)? };
        let route_capacity = global_seq_capacity * config.n_activated_experts;
        let route_weights = unsafe { ctx.stream.alloc(route_capacity)? };
        let route_indices = unsafe { ctx.stream.alloc(route_capacity)? };
        let local_experts = config.n_routed_experts / world_size;
        let pos_to_token = unsafe { ctx.stream.alloc(route_capacity)? };
        let pos_to_token_topk = unsafe { ctx.stream.alloc(route_capacity)? };
        let token_topk_to_pos = unsafe { ctx.stream.alloc(route_capacity)? };
        let expert_indptr = unsafe { ctx.stream.alloc(local_experts + 1)? };
        let expert_cursor = unsafe { ctx.stream.alloc(local_experts)? };
        let local_count = unsafe { ctx.stream.alloc(1)? };
        let expanded_input = Bf16HiddenStates::uninit(ctx, config.dim, route_capacity)?;
        let expert_gate = Bf16HiddenStates::uninit(ctx, config.moe_inter_dim, route_capacity)?;
        let expert_up = Bf16HiddenStates::uninit(ctx, config.moe_inter_dim, route_capacity)?;
        let expert_out = Bf16HiddenStates::uninit(ctx, config.dim, route_capacity)?;
        let max_fp4_input_dim = config.dim.max(config.moe_inter_dim);
        let max_fp4_scale_cols = max_fp4_input_dim.div_ceil(128);
        let fp4_act_workspace = unsafe { ctx.stream.alloc(route_capacity * max_fp4_input_dim)? };
        let fp4_act_scale_workspace =
            unsafe { ctx.stream.alloc(route_capacity * max_fp4_scale_cols)? };
        let partial_routed = F32HiddenStates {
            data: unsafe { ctx.stream.alloc(config.dim * global_seq_capacity)? },
            hidden_dim: config.dim,
            seq_len: global_seq_capacity,
        };
        let local_routed = F32HiddenStates {
            data: unsafe { ctx.stream.alloc(config.dim * local_seq_capacity)? },
            hidden_dim: config.dim,
            seq_len: local_seq_capacity,
        };
        let out = Bf16HiddenStates::uninit(ctx, config.dim, local_seq_capacity)?;
        Ok(Self {
            global_hidden,
            global_token_ids,
            route_weights,
            route_indices,
            pos_to_token,
            pos_to_token_topk,
            token_topk_to_pos,
            expert_indptr,
            expert_cursor,
            local_count,
            expanded_input,
            expert_gate,
            expert_up,
            expert_out,
            fp4_act_workspace,
            fp4_act_scale_workspace,
            partial_routed,
            local_routed,
            out,
        })
    }
}

#[cfg(feature = "pplx-ep")]
impl MoePplxScratch {
    pub(crate) fn new(
        ctx: &RankGpuContext,
        config: &Config,
        world_size: usize,
        local_seq_capacity: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            world_size > 0,
            "MoE pplx scratch world size must be positive"
        );
        ensure!(
            local_seq_capacity > 0,
            "MoE pplx scratch local capacity must be positive"
        );
        ensure!(
            config.n_routed_experts.is_multiple_of(world_size),
            "MoE pplx scratch n_routed_experts={} not divisible by world_size={world_size}",
            config.n_routed_experts
        );
        let topk = config.n_activated_experts;
        let local_experts = config.n_routed_experts / world_size;
        // Match the upstream pplx-garden formula (p2p_all_to_all.py:105).
        // The packed expert buffer is addressed via padded indices, so
        // capacity must include `expert_padding` slop. Keep these
        // parameters in sync with `pegainfer_comm::bootstrap::PplxBootstrapParams`.
        const EXPERT_PADDING: usize = 16;
        // Match PplxBootstrapParams::default().max_num_tokens — EpBackend
        // is initialized with this capacity, so the scratch buffer must
        // match.
        let max_num_tokens = std::cmp::max(local_seq_capacity, 8);
        let num_dp_groups = world_size;
        let num_tokens_total = max_num_tokens * num_dp_groups;
        let avg_tokens_per_expert = {
            let raw = (max_num_tokens * topk).div_ceil(config.n_routed_experts);
            raw + raw / 5 + 1
        };
        let max_private_tokens = avg_tokens_per_expert * local_experts;
        let round_up = |v: usize, m: usize| if m == 0 { v } else { v.div_ceil(m) * m };
        let max_recv_tokens = max_private_tokens * num_dp_groups
            + round_up(
                std::cmp::max(
                    std::cmp::min(
                        num_tokens_total * topk + local_experts * (EXPERT_PADDING - 1),
                        num_tokens_total * local_experts,
                    ),
                    local_experts * EXPERT_PADDING,
                ),
                EXPERT_PADDING,
            );

        let route_capacity = local_seq_capacity * topk;
        let route_weights = unsafe { ctx.stream.alloc(route_capacity)? };
        let route_indices = unsafe { ctx.stream.alloc(route_capacity)? };

        let expanded_input = Bf16HiddenStates::uninit(ctx, config.dim, max_recv_tokens)?;
        let expert_gate = Bf16HiddenStates::uninit(ctx, config.moe_inter_dim, max_recv_tokens)?;
        let expert_up = Bf16HiddenStates::uninit(ctx, config.moe_inter_dim, max_recv_tokens)?;
        let expert_out = Bf16HiddenStates::uninit(ctx, config.dim, max_recv_tokens)?;
        let max_fp4_input_dim = config.dim.max(config.moe_inter_dim);
        let max_fp4_scale_cols = max_fp4_input_dim.div_ceil(128);
        let fp4_act_workspace = unsafe { ctx.stream.alloc(max_recv_tokens * max_fp4_input_dim)? };
        let fp4_act_scale_workspace =
            unsafe { ctx.stream.alloc(max_recv_tokens * max_fp4_scale_cols)? };

        let expert_indptr = unsafe { ctx.stream.alloc(local_experts + 1)? };
        // dispatch_recv writes `out_num_tokens_ptr[expert]` for each local
        // expert (matches upstream Python `(num_local_experts,)` shape),
        // not a single scalar.
        let recv_tokens_per_expert = unsafe { ctx.stream.alloc(local_experts)? };

        let out = Bf16HiddenStates::uninit(ctx, config.dim, local_seq_capacity)?;

        Ok(Self {
            route_weights,
            route_indices,
            expanded_input,
            expert_gate,
            expert_up,
            expert_out,
            fp4_act_workspace,
            fp4_act_scale_workspace,
            expert_indptr,
            recv_tokens_per_expert,
            expert_padding: EXPERT_PADDING,
            out,
        })
    }
}

impl SharedExpertScratch {
    pub(crate) fn new(ctx: &RankGpuContext, config: &Config, seq_capacity: usize) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "shared expert scratch capacity must be positive"
        );
        let gate = Bf16HiddenStates::uninit(ctx, config.moe_inter_dim, seq_capacity)?;
        let up = Bf16HiddenStates::uninit(ctx, config.moe_inter_dim, seq_capacity)?;
        let out = Bf16HiddenStates::uninit(ctx, config.dim, seq_capacity)?;
        let max_fp8_input_dim = config.dim.max(config.moe_inter_dim);
        let max_fp8_scale_cols = max_fp8_input_dim.div_ceil(128);
        let fp8_act_workspace = unsafe { ctx.stream.alloc(seq_capacity * max_fp8_input_dim)? };
        let fp8_act_scale_workspace =
            unsafe { ctx.stream.alloc(seq_capacity * max_fp8_scale_cols)? };
        Ok(Self {
            gate,
            up,
            out,
            fp8_act_workspace,
            fp8_act_scale_workspace,
            seq_capacity,
        })
    }
}

impl AttentionOutputScratch {
    pub(crate) fn new(
        ctx: &RankGpuContext,
        config: &Config,
        world_size: usize,
        seq_capacity: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "attention output scratch capacity must be positive"
        );
        ensure!(
            world_size > 0,
            "attention output scratch world size must be positive"
        );
        ensure!(
            config.num_attention_heads.is_multiple_of(world_size),
            "attention output scratch num_attention_heads={} not divisible by world_size={}",
            config.num_attention_heads,
            world_size
        );
        ensure!(
            config.o_groups.is_multiple_of(world_size),
            "attention output scratch o_groups={} not divisible by world_size={}",
            config.o_groups,
            world_size
        );
        let local_heads = config.num_attention_heads / world_size;
        let local_groups = config.o_groups / world_size;
        let q_hidden_dim = local_heads * config.head_dim;
        let low_rank_dim = local_groups * config.o_lora_rank;
        let attn_out = Bf16HiddenStates::uninit(ctx, q_hidden_dim, seq_capacity)?;
        let low_rank = Bf16HiddenStates::uninit(ctx, low_rank_dim, seq_capacity)?;
        let out = Bf16HiddenStates::uninit(ctx, config.dim, seq_capacity)?;
        Ok(Self {
            attn_out,
            low_rank,
            out,
            local_heads,
            head_dim: config.head_dim,
            seq_capacity,
        })
    }
}

impl AttentionIndexScratch {
    pub(crate) fn new(ctx: &RankGpuContext, config: &Config, seq_capacity: usize) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            config.sliding_window > 0,
            "attention index scratch sliding_window must be positive"
        );
        ensure!(
            seq_capacity > 0,
            "attention index scratch seq_capacity must be positive"
        );
        let window_capacity = seq_capacity * config.sliding_window;
        let max_compressed_len = config.max_position_embeddings.div_ceil(4).max(1);
        let compress_capacity = seq_capacity * max_compressed_len;
        let compress_alloc = compress_capacity.max(1);
        let window_idxs = unsafe { ctx.stream.alloc(window_capacity)? };
        let compress_idxs = unsafe { ctx.stream.alloc(compress_alloc)? };
        let topk_idxs = unsafe { ctx.stream.alloc(window_capacity + compress_alloc)? };
        Ok(Self {
            window_idxs,
            compress_idxs,
            topk_idxs,
        })
    }
}

impl AttentionAuxScratch {
    pub(crate) fn new(
        ctx: &RankGpuContext,
        config: &Config,
        world_size: usize,
        seq_capacity: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            world_size > 0,
            "attention aux scratch world size must be positive"
        );
        ensure!(
            seq_capacity > 0,
            "attention aux scratch seq_capacity must be positive"
        );
        ensure!(
            config.index_n_heads.is_multiple_of(world_size),
            "attention aux scratch index_n_heads={} not divisible by world_size={}",
            config.index_n_heads,
            world_size
        );
        let max_head_dim = config.head_dim.max(config.index_head_dim);
        let local_index_heads = config.index_n_heads / world_size;
        let max_compressed_len = config.max_position_embeddings.div_ceil(4).max(1);
        let compressor_weighted = unsafe { ctx.stream.alloc(max_head_dim)? };
        let compressor_out = Bf16HiddenStates::uninit(ctx, max_head_dim, 1)?;
        let indexer_q =
            Bf16HiddenStates::uninit(ctx, local_index_heads * config.index_head_dim, seq_capacity)?;
        let indexer_weights = Bf16HiddenStates::uninit(ctx, local_index_heads, seq_capacity)?;
        let indexer_scores = unsafe { ctx.stream.alloc(seq_capacity * max_compressed_len)? };
        Ok(Self {
            compressor_weighted,
            compressor_out,
            indexer_q,
            indexer_weights,
            indexer_scores,
            max_head_dim,
            local_index_heads,
            max_compressed_len,
        })
    }
}

impl AttentionProjectionScratch {
    pub(crate) fn new(
        ctx: &RankGpuContext,
        config: &Config,
        world_size: usize,
        seq_capacity: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            seq_capacity > 0,
            "attention projection scratch capacity must be positive"
        );
        ensure!(
            world_size > 0,
            "attention projection scratch world size must be positive"
        );
        ensure!(
            config.num_attention_heads.is_multiple_of(world_size),
            "attention projection scratch num_attention_heads={} not divisible by world_size={}",
            config.num_attention_heads,
            world_size
        );
        let local_heads = config.num_attention_heads / world_size;
        let q_hidden_dim = local_heads * config.head_dim;
        let qr_raw = Bf16HiddenStates::uninit(ctx, config.q_lora_rank, seq_capacity)?;
        let qr = Bf16HiddenStates::uninit(ctx, config.q_lora_rank, seq_capacity)?;
        let q_raw = Bf16HiddenStates::uninit(ctx, q_hidden_dim, seq_capacity)?;
        let q = Bf16HiddenStates::uninit(ctx, q_hidden_dim, seq_capacity)?;
        let kv_raw = Bf16HiddenStates::uninit(ctx, config.head_dim, seq_capacity)?;
        let kv = Bf16HiddenStates::uninit(ctx, config.head_dim, seq_capacity)?;
        Ok(Self {
            qr_raw,
            qr,
            q_raw,
            q,
            kv_raw,
            kv,
            local_heads,
            head_dim: config.head_dim,
            seq_capacity,
        })
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
        Self::zeros_with_max_seq_and_slots(ctx, config, layer, max_seq_len, 1)
    }

    pub(crate) fn zeros_with_max_seq_and_slots(
        ctx: &RankGpuContext,
        config: &Config,
        layer: usize,
        max_seq_len: usize,
        request_slots: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        ensure!(
            layer < config.compress_ratios.len(),
            "decode cache layer {layer} out of range"
        );
        ensure!(max_seq_len > 0, "decode cache max_seq_len must be positive");
        ensure!(
            request_slots > 0,
            "decode cache request slots must be positive"
        );
        let ratio = config.compress_ratios[layer];
        let compressed_slots = if ratio > 0 {
            max_seq_len.div_ceil(ratio)
        } else {
            0
        };
        let kv = Bf16Cache::zeros(
            ctx,
            config.head_dim,
            request_slots * (config.sliding_window + compressed_slots),
        )?;
        let compressor = if ratio == 0 {
            None
        } else if ratio == 4 {
            Some(CompressorDecodeState::zeros(
                ctx,
                2 * config.head_dim,
                request_slots * 2 * ratio,
                f32::NEG_INFINITY,
            )?)
        } else {
            Some(CompressorDecodeState::zeros(
                ctx,
                config.head_dim,
                request_slots * ratio,
                f32::NEG_INFINITY,
            )?)
        };
        let indexer_kv = if ratio == 4 {
            Some(Bf16Cache::zeros(
                ctx,
                config.index_head_dim,
                request_slots * max_seq_len.div_ceil(ratio),
            )?)
        } else {
            None
        };
        let indexer_compressor = if ratio == 4 {
            Some(CompressorDecodeState::zeros(
                ctx,
                2 * config.index_head_dim,
                request_slots * 2 * ratio,
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

pub(crate) fn copy_bf16_rows_to_cache_indexed(
    ctx: &RankGpuContext,
    src: &Bf16HiddenStates,
    cache: &mut Bf16Cache,
    src_rows: &CudaSlice<i32>,
    dst_rows: &CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    ctx.set_current()?;
    ensure!(
        src.hidden_dim == cache.hidden_dim,
        "indexed copy rows hidden mismatch: src={}, cache={}",
        src.hidden_dim,
        cache.hidden_dim
    );
    ensure!(
        src_rows.len() >= rows,
        "indexed copy source rows capacity too small: need {}, have {}",
        rows,
        src_rows.len()
    );
    ensure!(
        dst_rows.len() >= rows,
        "indexed copy destination rows capacity too small: need {}, have {}",
        rows,
        dst_rows.len()
    );
    {
        let (src_ptr, _src_guard) = src.data.device_ptr(&ctx.stream);
        let (dst_ptr, _dst_guard) = cache.data.device_ptr_mut(&ctx.stream);
        let (src_rows_ptr, _src_rows_guard) = src_rows.device_ptr(&ctx.stream);
        let (dst_rows_ptr, _dst_rows_guard) = dst_rows.device_ptr(&ctx.stream);
        let result = unsafe {
            ffi::deepseek_bf16_copy_rows_indexed_cuda(
                src_ptr as *const ffi::Half,
                dst_ptr as *mut ffi::Half,
                src_rows_ptr as *const i32,
                dst_rows_ptr as *const i32,
                cache.hidden_dim as i32,
                rows as i32,
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

    pub fn uninit(
        ctx: &RankGpuContext,
        hidden_dim: usize,
        seq_len: usize,
        hc: usize,
    ) -> Result<Self> {
        ctx.set_current()?;
        let data = unsafe { ctx.stream.alloc(hidden_dim * seq_len * hc)? };
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

#[cfg(test)]
mod tests {
    use super::decode_cache_compressed_row;

    #[test]
    fn compressed_rows_follow_per_slot_window_then_compressed_layout() {
        let sliding_window = 128;
        let compressed_slots = 32;

        assert_eq!(
            decode_cache_compressed_row(sliding_window, compressed_slots, 0, 3),
            131
        );
        assert_eq!(
            decode_cache_compressed_row(sliding_window, compressed_slots, 1, 3),
            291
        );
    }
}
