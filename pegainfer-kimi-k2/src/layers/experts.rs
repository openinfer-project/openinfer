//! Kimi-K2.6 routed expert INT4 header/API draft.
//!
//! This module describes the compressed-tensors native INT4 expert surface for
//! the text-only Kimi-K2.6 path. It deliberately keeps CUDA/cuBLAS bodies out of
//! the temporary header crate while making the shape contract compile-checked.

use crate::{
    config::{
        KIMI_K2_EXPERT_INTERMEDIATE, KIMI_K2_HIDDEN, KIMI_K2_INT4_GROUP_SIZE,
        KIMI_K2_ROUTED_EXPERTS, KIMI_K2_TOPK, KimiK2ParallelShape,
    },
    tensor::{
        Bf16, DType, EpRank, F32, HeaderError, HeaderResult, Layout, Shape2, Shape3, StreamHandle,
        TensorMut, TensorRef, TokenBatch, U8, U32,
    },
};

pub const KIMI_K2_EP_WORLD: usize = 8;
pub const KIMI_K2_EP8_LOCAL_EXPERTS: usize = KIMI_K2_ROUTED_EXPERTS / KIMI_K2_EP_WORLD;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertLinearRole {
    W1Gate,
    W3Up,
    W2Down,
}

impl ExpertLinearRole {
    #[must_use]
    pub const fn expected_shape(self) -> Shape2 {
        match self {
            Self::W1Gate | Self::W3Up => Shape2 {
                rows: KIMI_K2_EXPERT_INTERMEDIATE,
                cols: KIMI_K2_HIDDEN,
            },
            Self::W2Down => Shape2 {
                rows: KIMI_K2_HIDDEN,
                cols: KIMI_K2_EXPERT_INTERMEDIATE,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int4NibbleOrder {
    LowThenHigh,
    HighThenLow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int4Encoding {
    SignedSymmetric,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedTensorsInt4Meta {
    pub role: ExpertLinearRole,
    pub global_experts: usize,
    pub local_experts: usize,
    pub local_expert_offset: usize,
    pub logical_shape: Shape2,
    pub packed_shape: Shape3,
    pub scale_shape: Shape3,
    pub weight_shape_entries: usize,
    pub group_size: usize,
    pub packed_dtype: DType,
    pub scale_dtype: DType,
    pub shape_dtype: DType,
    pub nibble_order: Int4NibbleOrder,
    pub encoding: Int4Encoding,
}

impl CompressedTensorsInt4Meta {
    #[must_use]
    pub fn ep8(role: ExpertLinearRole, ep_rank: EpRank, nibble_order: Int4NibbleOrder) -> Self {
        let logical_shape = role.expected_shape();
        let local_experts = KIMI_K2_EP8_LOCAL_EXPERTS;
        let packed_cols = packed_int4_cols(logical_shape.cols);
        let scale_cols = logical_shape.cols / KIMI_K2_INT4_GROUP_SIZE;

        Self {
            role,
            global_experts: KIMI_K2_ROUTED_EXPERTS,
            local_experts,
            local_expert_offset: ep_rank.rank * local_experts,
            logical_shape,
            packed_shape: Shape3 {
                outer: local_experts,
                middle: logical_shape.rows,
                inner: packed_cols,
            },
            scale_shape: Shape3 {
                outer: local_experts,
                middle: logical_shape.rows,
                inner: scale_cols,
            },
            weight_shape_entries: local_experts * 2,
            group_size: KIMI_K2_INT4_GROUP_SIZE,
            packed_dtype: DType::U8,
            scale_dtype: DType::Bf16,
            shape_dtype: DType::U32,
            nibble_order,
            encoding: Int4Encoding::SignedSymmetric,
        }
    }

    pub fn validate(&self) -> HeaderResult<()> {
        ensure(
            self.global_experts == KIMI_K2_ROUTED_EXPERTS,
            format!(
                "Kimi-K2.6 routed experts must be {KIMI_K2_ROUTED_EXPERTS}, got {}",
                self.global_experts
            ),
        )?;
        ensure(
            self.local_experts == KIMI_K2_EP8_LOCAL_EXPERTS,
            format!(
                "EP8 rank must own {KIMI_K2_EP8_LOCAL_EXPERTS} local experts, got {}",
                self.local_experts
            ),
        )?;
        ensure(
            self.local_expert_offset + self.local_experts <= self.global_experts,
            format!(
                "local expert range [{}..{}) exceeds {} global experts",
                self.local_expert_offset,
                self.local_expert_offset + self.local_experts,
                self.global_experts
            ),
        )?;
        ensure(
            self.logical_shape == self.role.expected_shape(),
            format!(
                "{:?} logical shape must be {:?}, got {:?}",
                self.role,
                self.role.expected_shape(),
                self.logical_shape
            ),
        )?;
        ensure(
            self.group_size == KIMI_K2_INT4_GROUP_SIZE,
            format!(
                "INT4 group size must be {KIMI_K2_INT4_GROUP_SIZE}, got {}",
                self.group_size
            ),
        )?;
        ensure(
            self.logical_shape.cols % self.group_size == 0,
            format!(
                "input dim {} must be divisible by group size {}",
                self.logical_shape.cols, self.group_size
            ),
        )?;
        ensure(
            self.packed_shape
                == Shape3 {
                    outer: self.local_experts,
                    middle: self.logical_shape.rows,
                    inner: packed_int4_cols(self.logical_shape.cols),
                },
            format!(
                "weight_packed shape must be {:?}, got {:?}",
                Shape3 {
                    outer: self.local_experts,
                    middle: self.logical_shape.rows,
                    inner: packed_int4_cols(self.logical_shape.cols),
                },
                self.packed_shape
            ),
        )?;
        ensure(
            self.scale_shape
                == Shape3 {
                    outer: self.local_experts,
                    middle: self.logical_shape.rows,
                    inner: self.logical_shape.cols / self.group_size,
                },
            format!(
                "weight_scale shape must be {:?}, got {:?}",
                Shape3 {
                    outer: self.local_experts,
                    middle: self.logical_shape.rows,
                    inner: self.logical_shape.cols / self.group_size,
                },
                self.scale_shape
            ),
        )?;
        ensure(
            self.weight_shape_entries == self.local_experts * 2,
            format!(
                "weight_shape must carry [out, in] for each local expert: expected {} u32 entries, got {}",
                self.local_experts * 2,
                self.weight_shape_entries
            ),
        )?;
        ensure(
            self.packed_dtype == DType::U8,
            format!(
                "weight_packed dtype must be U8, got {:?}",
                self.packed_dtype
            ),
        )?;
        ensure(
            self.scale_dtype == DType::Bf16,
            format!(
                "weight_scale dtype must be BF16, got {:?}",
                self.scale_dtype
            ),
        )?;
        ensure(
            self.shape_dtype == DType::U32,
            format!("weight_shape dtype must be U32, got {:?}", self.shape_dtype),
        )?;
        ensure(
            self.encoding == Int4Encoding::SignedSymmetric,
            format!(
                "only signed symmetric INT4 is currently specified, got {:?}",
                self.encoding
            ),
        )?;

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4PackedLinear {
    pub meta: CompressedTensorsInt4Meta,
    pub weight_packed: TensorRef<U8>,
    pub weight_scale: TensorRef<Bf16>,
    pub weight_shape: TensorRef<U32>,
}

impl Int4PackedLinear {
    #[must_use]
    pub const fn new(
        meta: CompressedTensorsInt4Meta,
        weight_packed: TensorRef<U8>,
        weight_scale: TensorRef<Bf16>,
        weight_shape: TensorRef<U32>,
    ) -> Self {
        Self {
            meta,
            weight_packed,
            weight_scale,
            weight_shape,
        }
    }

    pub fn validate(&self) -> HeaderResult<()> {
        self.meta.validate()?;
        ensure(
            self.weight_packed.dtype == self.meta.packed_dtype,
            format!(
                "{:?} weight_packed dtype mismatch: meta {:?}, tensor {:?}",
                self.meta.role, self.meta.packed_dtype, self.weight_packed.dtype
            ),
        )?;
        ensure(
            self.weight_packed.layout == Layout::ExpertMajor,
            format!(
                "{:?} weight_packed layout must be ExpertMajor, got {:?}",
                self.meta.role, self.weight_packed.layout
            ),
        )?;
        ensure(
            self.weight_packed.ptr.len == shape3_elems(self.meta.packed_shape),
            format!(
                "{:?} weight_packed len must be {}, got {}",
                self.meta.role,
                shape3_elems(self.meta.packed_shape),
                self.weight_packed.ptr.len
            ),
        )?;
        ensure(
            self.weight_scale.dtype == self.meta.scale_dtype,
            format!(
                "{:?} weight_scale dtype mismatch: meta {:?}, tensor {:?}",
                self.meta.role, self.meta.scale_dtype, self.weight_scale.dtype
            ),
        )?;
        ensure(
            self.weight_scale.layout == Layout::ExpertMajor,
            format!(
                "{:?} weight_scale layout must be ExpertMajor, got {:?}",
                self.meta.role, self.weight_scale.layout
            ),
        )?;
        ensure(
            self.weight_scale.ptr.len == shape3_elems(self.meta.scale_shape),
            format!(
                "{:?} weight_scale len must be {}, got {}",
                self.meta.role,
                shape3_elems(self.meta.scale_shape),
                self.weight_scale.ptr.len
            ),
        )?;
        ensure(
            self.weight_shape.dtype == self.meta.shape_dtype,
            format!(
                "{:?} weight_shape dtype mismatch: meta {:?}, tensor {:?}",
                self.meta.role, self.meta.shape_dtype, self.weight_shape.dtype
            ),
        )?;
        ensure(
            self.weight_shape.ptr.len == self.meta.weight_shape_entries,
            format!(
                "{:?} weight_shape len must be {}, got {}",
                self.meta.role, self.meta.weight_shape_entries, self.weight_shape.ptr.len
            ),
        )?;

        Ok(())
    }

    #[must_use]
    pub const fn dequant_bf16_elements(&self) -> usize {
        self.meta.local_experts * self.meta.logical_shape.rows * self.meta.logical_shape.cols
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedExpertWeights {
    pub ep_rank: EpRank,
    pub parallel: KimiK2ParallelShape,
    pub w1_gate: Int4PackedLinear,
    pub w3_up: Int4PackedLinear,
    pub w2_down: Int4PackedLinear,
}

impl GroupedExpertWeights {
    pub fn new_ep8(
        ep_rank: EpRank,
        w1_gate: Int4PackedLinear,
        w3_up: Int4PackedLinear,
        w2_down: Int4PackedLinear,
    ) -> HeaderResult<Self> {
        ensure(
            ep_rank.world == KIMI_K2_EP_WORLD,
            format!(
                "Kimi-K2.6 first EP contract is EP8, got EP{}",
                ep_rank.world
            ),
        )?;
        let parallel = KimiK2ParallelShape::tp8_ep8();
        let weights = Self {
            ep_rank,
            parallel,
            w1_gate,
            w3_up,
            w2_down,
        };
        weights.validate()?;
        Ok(weights)
    }

    pub fn validate(&self) -> HeaderResult<()> {
        self.w1_gate.validate()?;
        self.w3_up.validate()?;
        self.w2_down.validate()?;
        ensure_role(self.w1_gate.meta.role, ExpertLinearRole::W1Gate)?;
        ensure_role(self.w3_up.meta.role, ExpertLinearRole::W3Up)?;
        ensure_role(self.w2_down.meta.role, ExpertLinearRole::W2Down)?;

        let expected_offset = self.ep_rank.rank * self.parallel.local_experts;
        for linear in [self.w1_gate, self.w3_up, self.w2_down] {
            ensure(
                linear.meta.local_experts == self.parallel.local_experts,
                format!(
                    "{:?} local experts must match parallel shape {}",
                    linear.meta.role, self.parallel.local_experts
                ),
            )?;
            ensure(
                linear.meta.local_expert_offset == expected_offset,
                format!(
                    "{:?} local expert offset must be {}, got {}",
                    linear.meta.role, expected_offset, linear.meta.local_expert_offset
                ),
            )?;
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertRouteLayout {
    pub batch: TokenBatch,
    pub local_expert_offset: usize,
    pub local_experts: usize,
    pub routed_tokens: usize,
    pub topk_indices: TensorRef<U32>,
    pub topk_weights: TensorRef<F32>,
    pub expert_indptr: TensorMut<U32>,
    pub route_to_token: TensorMut<U32>,
    pub route_to_topk_slot: TensorMut<U32>,
}

impl ExpertRouteLayout {
    #[must_use]
    pub const fn max_routed_tokens(active_tokens: usize) -> usize {
        active_tokens * KIMI_K2_TOPK
    }

    pub fn validate(&self) -> HeaderResult<()> {
        ensure(
            self.batch.batch_size > 0,
            "batch_size must be > 0 for routed experts".to_owned(),
        )?;
        ensure(
            self.batch.active_tokens > 0,
            "active_tokens must be > 0 for routed experts".to_owned(),
        )?;
        ensure(
            self.batch.padded_tokens >= self.batch.active_tokens,
            format!(
                "padded_tokens {} must cover active_tokens {}",
                self.batch.padded_tokens, self.batch.active_tokens
            ),
        )?;
        ensure(
            self.local_experts == KIMI_K2_EP8_LOCAL_EXPERTS,
            format!(
                "expert route layout must target {KIMI_K2_EP8_LOCAL_EXPERTS} local experts, got {}",
                self.local_experts
            ),
        )?;
        ensure(
            self.routed_tokens <= Self::max_routed_tokens(self.batch.active_tokens),
            format!(
                "routed_tokens {} exceeds active_tokens * topk {}",
                self.routed_tokens,
                Self::max_routed_tokens(self.batch.active_tokens)
            ),
        )?;
        ensure(
            self.topk_indices.dtype == DType::U32 && self.topk_indices.layout == Layout::RowMajor,
            format!(
                "topk_indices must be row-major U32 [tokens, {KIMI_K2_TOPK}], got {:?}/{:?}",
                self.topk_indices.dtype, self.topk_indices.layout
            ),
        )?;
        ensure(
            self.topk_indices.ptr.len >= self.batch.active_tokens * KIMI_K2_TOPK,
            format!(
                "topk_indices len must cover {}, got {}",
                self.batch.active_tokens * KIMI_K2_TOPK,
                self.topk_indices.ptr.len
            ),
        )?;
        ensure(
            self.topk_weights.dtype == DType::F32 && self.topk_weights.layout == Layout::RowMajor,
            format!(
                "topk_weights must be row-major F32 [tokens, {KIMI_K2_TOPK}], got {:?}/{:?}",
                self.topk_weights.dtype, self.topk_weights.layout
            ),
        )?;
        ensure(
            self.topk_weights.ptr.len >= self.batch.active_tokens * KIMI_K2_TOPK,
            format!(
                "topk_weights len must cover {}, got {}",
                self.batch.active_tokens * KIMI_K2_TOPK,
                self.topk_weights.ptr.len
            ),
        )?;
        ensure(
            self.expert_indptr.dtype == DType::U32
                && self.expert_indptr.layout == Layout::ExpertMajor,
            format!(
                "expert_indptr must be ExpertMajor U32 [local_experts + 1], got {:?}/{:?}",
                self.expert_indptr.dtype, self.expert_indptr.layout
            ),
        )?;
        ensure(
            self.expert_indptr.ptr.len >= self.local_experts + 1,
            format!(
                "expert_indptr len must cover {}, got {}",
                self.local_experts + 1,
                self.expert_indptr.ptr.len
            ),
        )?;
        ensure_route_map("route_to_token", self.route_to_token, self.routed_tokens)?;
        ensure_route_map(
            "route_to_topk_slot",
            self.route_to_topk_slot,
            self.routed_tokens,
        )?;

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertScratch {
    pub max_batch_size: usize,
    pub max_active_tokens: usize,
    pub max_routed_tokens: usize,
    pub expert_indptr: TensorMut<U32>,
    pub route_to_token: TensorMut<U32>,
    pub route_to_topk_slot: TensorMut<U32>,
    pub expert_hidden: TensorMut<Bf16>,
    pub w1_gate: TensorMut<Bf16>,
    pub w3_up: TensorMut<Bf16>,
    pub swiglu: TensorMut<Bf16>,
    pub expert_output: TensorMut<Bf16>,
    pub reduced_output: TensorMut<F32>,
}

impl ExpertScratch {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        max_batch_size: usize,
        max_active_tokens: usize,
        expert_indptr: TensorMut<U32>,
        route_to_token: TensorMut<U32>,
        route_to_topk_slot: TensorMut<U32>,
        expert_hidden: TensorMut<Bf16>,
        w1_gate: TensorMut<Bf16>,
        w3_up: TensorMut<Bf16>,
        swiglu: TensorMut<Bf16>,
        expert_output: TensorMut<Bf16>,
        reduced_output: TensorMut<F32>,
    ) -> Self {
        Self {
            max_batch_size,
            max_active_tokens,
            max_routed_tokens: max_active_tokens * KIMI_K2_TOPK,
            expert_indptr,
            route_to_token,
            route_to_topk_slot,
            expert_hidden,
            w1_gate,
            w3_up,
            swiglu,
            expert_output,
            reduced_output,
        }
    }

    pub fn validate(&self) -> HeaderResult<()> {
        ensure(
            self.max_batch_size > 0,
            "scratch max_batch_size must be > 0".to_owned(),
        )?;
        ensure(
            self.max_active_tokens > 0,
            "scratch max_active_tokens must be > 0".to_owned(),
        )?;
        ensure(
            self.max_routed_tokens == self.max_active_tokens * KIMI_K2_TOPK,
            format!(
                "max_routed_tokens must equal max_active_tokens * {KIMI_K2_TOPK}: got {} for {} active tokens",
                self.max_routed_tokens, self.max_active_tokens
            ),
        )?;
        ensure_mut(
            "scratch.expert_indptr",
            self.expert_indptr,
            DType::U32,
            Layout::ExpertMajor,
            KIMI_K2_EP8_LOCAL_EXPERTS + 1,
        )?;
        ensure_mut(
            "scratch.route_to_token",
            self.route_to_token,
            DType::U32,
            Layout::ExpertMajor,
            self.max_routed_tokens,
        )?;
        ensure_mut(
            "scratch.route_to_topk_slot",
            self.route_to_topk_slot,
            DType::U32,
            Layout::ExpertMajor,
            self.max_routed_tokens,
        )?;
        ensure_mut(
            "scratch.expert_hidden",
            self.expert_hidden,
            DType::Bf16,
            Layout::ExpertMajor,
            self.max_routed_tokens * KIMI_K2_HIDDEN,
        )?;
        ensure_mut(
            "scratch.w1_gate",
            self.w1_gate,
            DType::Bf16,
            Layout::ExpertMajor,
            self.max_routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_mut(
            "scratch.w3_up",
            self.w3_up,
            DType::Bf16,
            Layout::ExpertMajor,
            self.max_routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_mut(
            "scratch.swiglu",
            self.swiglu,
            DType::Bf16,
            Layout::ExpertMajor,
            self.max_routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_mut(
            "scratch.expert_output",
            self.expert_output,
            DType::Bf16,
            Layout::ExpertMajor,
            self.max_routed_tokens * KIMI_K2_HIDDEN,
        )?;
        ensure_mut(
            "scratch.reduced_output",
            self.reduced_output,
            DType::F32,
            Layout::RowMajor,
            self.max_active_tokens * KIMI_K2_HIDDEN,
        )?;

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedW1W3Plan {
    pub route: ExpertRouteLayout,
    pub hidden: TensorRef<Bf16>,
    pub expert_hidden: TensorMut<Bf16>,
    pub gate_out: TensorMut<Bf16>,
    pub up_out: TensorMut<Bf16>,
}

impl GroupedW1W3Plan {
    pub fn validate(&self, weights: &GroupedExpertWeights) -> HeaderResult<()> {
        weights.validate()?;
        self.route.validate()?;
        ensure_ref(
            "w1w3.hidden",
            self.hidden,
            DType::Bf16,
            Layout::RowMajor,
            self.route.batch.active_tokens * KIMI_K2_HIDDEN,
        )?;
        ensure_mut(
            "w1w3.expert_hidden",
            self.expert_hidden,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_HIDDEN,
        )?;
        ensure_mut(
            "w1w3.gate_out",
            self.gate_out,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_mut(
            "w1w3.up_out",
            self.up_out,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedW2SwiGLUPlan {
    pub route: ExpertRouteLayout,
    pub gate: TensorRef<Bf16>,
    pub up: TensorRef<Bf16>,
    pub swiglu: TensorMut<Bf16>,
    pub expert_output: TensorMut<Bf16>,
    pub reduced_output: TensorMut<F32>,
}

impl GroupedW2SwiGLUPlan {
    pub fn validate(&self, weights: &GroupedExpertWeights) -> HeaderResult<()> {
        weights.validate()?;
        self.route.validate()?;
        ensure_ref(
            "w2_swiglu.gate",
            self.gate,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_ref(
            "w2_swiglu.up",
            self.up,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_mut(
            "w2_swiglu.swiglu",
            self.swiglu,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_EXPERT_INTERMEDIATE,
        )?;
        ensure_mut(
            "w2_swiglu.expert_output",
            self.expert_output,
            DType::Bf16,
            Layout::ExpertMajor,
            self.route.routed_tokens * KIMI_K2_HIDDEN,
        )?;
        ensure_mut(
            "w2_swiglu.reduced_output",
            self.reduced_output,
            DType::F32,
            Layout::RowMajor,
            self.route.batch.active_tokens * KIMI_K2_HIDDEN,
        )?;

        Ok(())
    }
}

pub fn compressed_tensors_int4_header_probe(
    linear: &Int4PackedLinear,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    linear.validate()
}

pub fn kimi_int4_weight_loader(
    ep_rank: EpRank,
    w1_gate: Int4PackedLinear,
    w3_up: Int4PackedLinear,
    w2_down: Int4PackedLinear,
    _stream: StreamHandle,
) -> HeaderResult<GroupedExpertWeights> {
    GroupedExpertWeights::new_ep8(ep_rank, w1_gate, w3_up, w2_down)
}

pub fn moe_count_local_experts(
    route: &ExpertRouteLayout,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    route.validate()?;
    unsupported("moe_count_local_experts CUDA body is not part of this header crate")
}

pub fn moe_expert_indptr_prefix(
    route: &ExpertRouteLayout,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    route.validate()?;
    unsupported("moe_expert_indptr_prefix CUDA body is not part of this header crate")
}

pub fn moe_expand_to_expert_major(
    plan: &GroupedW1W3Plan,
    weights: &GroupedExpertWeights,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    plan.validate(weights)?;
    unsupported("moe_expand_to_expert_major CUDA body is not part of this header crate")
}

pub fn int4_dequant_bf16_format_probe(
    linear: &Int4PackedLinear,
    dequantized_weight: TensorMut<Bf16>,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    linear.validate()?;
    ensure_mut(
        "dequantized_weight",
        dequantized_weight,
        DType::Bf16,
        Layout::ExpertMajor,
        linear.dequant_bf16_elements(),
    )?;
    unsupported("INT4 dequant-to-BF16 format probe body is not part of this header crate")
}

pub fn int4_dequant_bf16_format_probe_w1_w3(
    plan: &GroupedW1W3Plan,
    weights: &GroupedExpertWeights,
    dequant_w1_gate: TensorMut<Bf16>,
    dequant_w3_up: TensorMut<Bf16>,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    plan.validate(weights)?;
    ensure_mut(
        "dequant_w1_gate",
        dequant_w1_gate,
        DType::Bf16,
        Layout::ExpertMajor,
        weights.w1_gate.dequant_bf16_elements(),
    )?;
    ensure_mut(
        "dequant_w3_up",
        dequant_w3_up,
        DType::Bf16,
        Layout::ExpertMajor,
        weights.w3_up.dequant_bf16_elements(),
    )?;
    unsupported("W1/W3 dequant-to-BF16 format probe body is not part of this header crate")
}

pub fn int4_dequant_bf16_format_probe_w2_swiglu(
    plan: &GroupedW2SwiGLUPlan,
    weights: &GroupedExpertWeights,
    dequant_w2_down: TensorMut<Bf16>,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    plan.validate(weights)?;
    ensure_mut(
        "dequant_w2_down",
        dequant_w2_down,
        DType::Bf16,
        Layout::ExpertMajor,
        weights.w2_down.dequant_bf16_elements(),
    )?;
    unsupported("W2 SwiGLU dequant-to-BF16 format probe body is not part of this header crate")
}

pub fn int4_grouped_w1_w3(
    plan: &GroupedW1W3Plan,
    weights: &GroupedExpertWeights,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    plan.validate(weights)?;
    unsupported("fused grouped W1/W3 INT4 kernel body is not part of this header crate")
}

pub fn int4_grouped_w2_swiglu(
    plan: &GroupedW2SwiGLUPlan,
    weights: &GroupedExpertWeights,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    plan.validate(weights)?;
    unsupported("fused grouped W2 SwiGLU INT4 kernel body is not part of this header crate")
}

pub fn moe_reduce_expert_outputs(
    plan: &GroupedW2SwiGLUPlan,
    weights: &GroupedExpertWeights,
    _stream: StreamHandle,
) -> HeaderResult<()> {
    plan.validate(weights)?;
    unsupported("moe_reduce_expert_outputs CUDA body is not part of this header crate")
}

#[must_use]
pub const fn packed_int4_cols(cols: usize) -> usize {
    cols.div_ceil(2)
}

#[must_use]
pub const fn shape3_elems(shape: Shape3) -> usize {
    shape.outer * shape.middle * shape.inner
}

fn ensure(condition: bool, message: String) -> HeaderResult<()> {
    if condition {
        Ok(())
    } else {
        Err(HeaderError::Shape { message })
    }
}

fn ensure_role(actual: ExpertLinearRole, expected: ExpertLinearRole) -> HeaderResult<()> {
    ensure(
        actual == expected,
        format!(
            "expert linear role must be {:?}, got {:?}",
            expected, actual
        ),
    )
}

fn ensure_ref<T>(
    name: &str,
    tensor: TensorRef<T>,
    dtype: DType,
    layout: Layout,
    min_len: usize,
) -> HeaderResult<()> {
    ensure(
        tensor.dtype == dtype && tensor.layout == layout,
        format!(
            "{name} must be {:?}/{:?}, got {:?}/{:?}",
            dtype, layout, tensor.dtype, tensor.layout
        ),
    )?;
    ensure(
        tensor.ptr.len >= min_len,
        format!("{name} len must cover {min_len}, got {}", tensor.ptr.len),
    )
}

fn ensure_mut<T>(
    name: &str,
    tensor: TensorMut<T>,
    dtype: DType,
    layout: Layout,
    min_len: usize,
) -> HeaderResult<()> {
    ensure(
        tensor.dtype == dtype && tensor.layout == layout,
        format!(
            "{name} must be {:?}/{:?}, got {:?}/{:?}",
            dtype, layout, tensor.dtype, tensor.layout
        ),
    )?;
    ensure(
        tensor.ptr.len >= min_len,
        format!("{name} len must cover {min_len}, got {}", tensor.ptr.len),
    )
}

fn ensure_route_map(name: &str, tensor: TensorMut<U32>, min_len: usize) -> HeaderResult<()> {
    ensure_mut(name, tensor, DType::U32, Layout::ExpertMajor, min_len)
}

fn unsupported<T>(message: &str) -> HeaderResult<T> {
    Err(HeaderError::Unsupported {
        message: message.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DevicePtr;

    #[test]
    fn ep8_meta_matches_kimi_shapes() {
        let ep_rank = EpRank { rank: 3, world: 8 };
        let w1 = CompressedTensorsInt4Meta::ep8(
            ExpertLinearRole::W1Gate,
            ep_rank,
            Int4NibbleOrder::LowThenHigh,
        );
        assert_eq!(w1.local_experts, 48);
        assert_eq!(w1.local_expert_offset, 144);
        assert_eq!(w1.logical_shape.rows, 2048);
        assert_eq!(w1.logical_shape.cols, 7168);
        assert_eq!(w1.packed_shape.inner, 3584);
        assert_eq!(w1.scale_shape.inner, 224);
        w1.validate().unwrap();

        let w2 = CompressedTensorsInt4Meta::ep8(
            ExpertLinearRole::W2Down,
            ep_rank,
            Int4NibbleOrder::LowThenHigh,
        );
        assert_eq!(w2.logical_shape.rows, 7168);
        assert_eq!(w2.logical_shape.cols, 2048);
        assert_eq!(w2.packed_shape.inner, 1024);
        assert_eq!(w2.scale_shape.inner, 64);
        w2.validate().unwrap();
    }

    #[test]
    fn route_layout_allows_multi_sequence_batches() {
        let route = ExpertRouteLayout {
            batch: TokenBatch {
                batch_size: 4,
                active_tokens: 17,
                padded_tokens: 32,
            },
            local_expert_offset: 0,
            local_experts: 48,
            routed_tokens: 17 * KIMI_K2_TOPK,
            topk_indices: tref_u32(17 * KIMI_K2_TOPK, Layout::RowMajor),
            topk_weights: tref_f32(17 * KIMI_K2_TOPK, Layout::RowMajor),
            expert_indptr: tmut_u32(49, Layout::ExpertMajor),
            route_to_token: tmut_u32(17 * KIMI_K2_TOPK, Layout::ExpertMajor),
            route_to_topk_slot: tmut_u32(17 * KIMI_K2_TOPK, Layout::ExpertMajor),
        };

        route.validate().unwrap();
    }

    fn tref_u32(len: usize, layout: Layout) -> TensorRef<U32> {
        TensorRef {
            ptr: DevicePtr::new(1, len),
            dtype: DType::U32,
            layout,
        }
    }

    fn tref_f32(len: usize, layout: Layout) -> TensorRef<F32> {
        TensorRef {
            ptr: DevicePtr::new(1, len),
            dtype: DType::F32,
            layout,
        }
    }

    fn tmut_u32(len: usize, layout: Layout) -> TensorMut<U32> {
        TensorMut {
            ptr: DevicePtr::new(1, len),
            dtype: DType::U32,
            layout,
        }
    }
}
