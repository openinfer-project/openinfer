#![allow(dead_code, unreachable_pub)]

use anyhow::{Result, bail, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi;
use crate::tensor::{AxisSpec, DeviceContext, HiddenStates, KernelCall, TensorSpec};

pub const KIMI_K2_HIDDEN: usize = 7168;
pub const KIMI_K2_EXPERT_INTERMEDIATE: usize = 2048;
pub const KIMI_K2_ROUTED_EXPERTS: usize = 384;
pub const KIMI_K2_EP_WORLD: usize = 8;
pub const KIMI_K2_LOCAL_EXPERTS: usize = KIMI_K2_ROUTED_EXPERTS / KIMI_K2_EP_WORLD;
pub const KIMI_K2_TOPK: usize = 8;
pub const KIMI_K2_INT4_GROUP_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiCutlassSm90aSupport {
    pub supported: bool,
    pub sm_major: i32,
    pub sm_minor: i32,
}

pub struct KimiCutlassInt4GroupedWorkspace {
    pub storage: CudaSlice<u8>,
    pub sizes: ffi::KimiCutlassInt4GroupedWorkspaceSizes,
    pub max_routed_tokens: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub local_experts: usize,
    pub group_size: usize,
}

impl KimiCutlassInt4GroupedWorkspace {
    pub fn new(
        ctx: &DeviceContext,
        max_routed_tokens: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let sizes = kimi_cutlass_int4_grouped_workspace_sizes(max_routed_tokens, in_dim, out_dim)?;
        ensure!(
            sizes.total_bytes > 0,
            "CUTLASS INT4 grouped workspace size query returned zero bytes"
        );
        let storage = ctx.stream.alloc_zeros(sizes.total_bytes)?;
        Ok(Self {
            storage,
            sizes,
            max_routed_tokens,
            in_dim,
            out_dim,
            local_experts: KIMI_K2_LOCAL_EXPERTS,
            group_size: KIMI_K2_INT4_GROUP_SIZE,
        })
    }

    pub fn validate_for(&self, routed_tokens: usize, in_dim: usize, out_dim: usize) -> Result<()> {
        ensure!(
            routed_tokens <= self.max_routed_tokens,
            "routed_tokens {} exceeds CUTLASS workspace capacity {}",
            routed_tokens,
            self.max_routed_tokens
        );
        ensure!(
            self.in_dim == in_dim && self.out_dim == out_dim,
            "CUTLASS workspace dims must be in={} out={}, got in={} out={}",
            in_dim,
            out_dim,
            self.in_dim,
            self.out_dim
        );
        ensure!(
            self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "CUTLASS workspace local_experts must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.local_experts
        );
        ensure!(
            self.group_size == KIMI_K2_INT4_GROUP_SIZE,
            "CUTLASS workspace group_size must be {}, got {}",
            KIMI_K2_INT4_GROUP_SIZE,
            self.group_size
        );
        ensure!(
            self.storage.len() >= self.sizes.total_bytes,
            "CUTLASS workspace storage len must cover {} bytes, got {}",
            self.sizes.total_bytes,
            self.storage.len()
        );
        Ok(())
    }

    #[must_use]
    pub fn contract_call(&self) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.int4_cutlass_grouped_workspace",
            "Kimi-K2 CUTLASS INT4 grouped GEMM workspace contract",
        )
        .output(
            "problem_sizes",
            TensorSpec::named(
                "opaque",
                "device_resident_cutlass_group_problem_shape",
                [AxisSpec::named("local_expert", self.local_experts)],
            ),
        )
        .output(
            "ptr_arrays",
            TensorSpec::named(
                "opaque",
                "device_resident_cutlass_ptr_arrays",
                [AxisSpec::named("local_expert", self.local_experts)],
            ),
        )
        .output(
            "stride_layout_arrays",
            TensorSpec::named(
                "opaque",
                "device_resident_cutlass_stride_layout_arrays",
                [AxisSpec::named("local_expert", self.local_experts)],
            ),
        )
        .attr(
            "problem_sizes_bytes",
            self.sizes.problem_sizes_bytes.to_string(),
        )
        .attr("ptr_arrays_bytes", self.sizes.ptr_arrays_bytes.to_string())
        .attr(
            "stride_arrays_bytes",
            self.sizes.stride_arrays_bytes.to_string(),
        )
        .attr(
            "layout_arrays_bytes",
            self.sizes.layout_arrays_bytes.to_string(),
        )
        .attr(
            "cutlass_workspace_bytes",
            self.sizes.cutlass_workspace_bytes.to_string(),
        )
        .attr("total_bytes", self.sizes.total_bytes.to_string())
        .attr("alignment", self.sizes.alignment.to_string())
        .attr("device_resident_problem_sizes", "true".to_string())
        .attr("device_resident_ptr_arrays", "true".to_string())
        .attr("decode_step_allocation", "forbidden".to_string())
        .attr("decode_step_d2h", "forbidden".to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiInt4ExpertRole {
    W1Gate,
    W3Up,
    W2Down,
}

impl KimiInt4ExpertRole {
    #[must_use]
    pub const fn expected_shape(self) -> KimiInt4LogicalShape {
        match self {
            Self::W1Gate | Self::W3Up => KimiInt4LogicalShape {
                out_dim: KIMI_K2_EXPERT_INTERMEDIATE,
                in_dim: KIMI_K2_HIDDEN,
            },
            Self::W2Down => KimiInt4LogicalShape {
                out_dim: KIMI_K2_HIDDEN,
                in_dim: KIMI_K2_EXPERT_INTERMEDIATE,
            },
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::W1Gate => "w1_gate",
            Self::W3Up => "w3_up",
            Self::W2Down => "w2_down",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiInt4NibbleOrder {
    LowThenHigh,
    HighThenLow,
}

impl KimiInt4NibbleOrder {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LowThenHigh => "low_then_high",
            Self::HighThenLow => "high_then_low",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiInt4Encoding {
    SignedSymmetric,
}

impl KimiInt4Encoding {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SignedSymmetric => "signed_symmetric",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiInt4LogicalShape {
    pub out_dim: usize,
    pub in_dim: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiInt4TensorShape {
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
}

impl KimiInt4TensorShape {
    #[must_use]
    pub const fn elements(self) -> usize {
        self.experts * self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiInt4WeightManifest {
    pub role: KimiInt4ExpertRole,
    pub global_experts: usize,
    pub local_experts: usize,
    pub local_expert_offset: usize,
    pub logical_shape: KimiInt4LogicalShape,
    pub packed_shape: KimiInt4TensorShape,
    pub scale_shape: KimiInt4TensorShape,
    pub weight_shape_entries: usize,
    pub group_size: usize,
    pub nibble_order: KimiInt4NibbleOrder,
    pub encoding: KimiInt4Encoding,
}

impl KimiInt4WeightManifest {
    #[must_use]
    pub fn ep8(
        role: KimiInt4ExpertRole,
        ep_rank: usize,
        nibble_order: KimiInt4NibbleOrder,
    ) -> Self {
        let logical_shape = role.expected_shape();
        let local_expert_offset = ep_rank * KIMI_K2_LOCAL_EXPERTS;
        Self {
            role,
            global_experts: KIMI_K2_ROUTED_EXPERTS,
            local_experts: KIMI_K2_LOCAL_EXPERTS,
            local_expert_offset,
            logical_shape,
            packed_shape: KimiInt4TensorShape {
                experts: KIMI_K2_LOCAL_EXPERTS,
                rows: logical_shape.out_dim,
                cols: packed_int4_cols(logical_shape.in_dim),
            },
            scale_shape: KimiInt4TensorShape {
                experts: KIMI_K2_LOCAL_EXPERTS,
                rows: logical_shape.out_dim,
                cols: logical_shape.in_dim / KIMI_K2_INT4_GROUP_SIZE,
            },
            weight_shape_entries: KIMI_K2_LOCAL_EXPERTS * 2,
            group_size: KIMI_K2_INT4_GROUP_SIZE,
            nibble_order,
            encoding: KimiInt4Encoding::SignedSymmetric,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.global_experts == KIMI_K2_ROUTED_EXPERTS,
            "Kimi-K2 routed experts must be {}, got {}",
            KIMI_K2_ROUTED_EXPERTS,
            self.global_experts
        );
        ensure!(
            self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Kimi-K2 EP8 rank must own {} local experts, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.local_experts
        );
        ensure!(
            self.local_expert_offset + self.local_experts <= self.global_experts,
            "local expert range [{}..{}) exceeds {} global experts",
            self.local_expert_offset,
            self.local_expert_offset + self.local_experts,
            self.global_experts
        );
        ensure!(
            self.logical_shape == self.role.expected_shape(),
            "{} logical shape must be {:?}, got {:?}",
            self.role.label(),
            self.role.expected_shape(),
            self.logical_shape
        );
        ensure!(
            self.group_size == KIMI_K2_INT4_GROUP_SIZE,
            "Kimi-K2 compressed-tensors INT4 group size must be {}, got {}",
            KIMI_K2_INT4_GROUP_SIZE,
            self.group_size
        );
        ensure!(
            self.logical_shape.in_dim.is_multiple_of(self.group_size),
            "input dim {} must be divisible by group size {}",
            self.logical_shape.in_dim,
            self.group_size
        );

        let expected_packed = KimiInt4TensorShape {
            experts: self.local_experts,
            rows: self.logical_shape.out_dim,
            cols: packed_int4_cols(self.logical_shape.in_dim),
        };
        ensure!(
            self.packed_shape == expected_packed,
            "{} weight_packed shape must be {:?}, got {:?}",
            self.role.label(),
            expected_packed,
            self.packed_shape
        );

        let expected_scale = KimiInt4TensorShape {
            experts: self.local_experts,
            rows: self.logical_shape.out_dim,
            cols: self.logical_shape.in_dim / self.group_size,
        };
        ensure!(
            self.scale_shape == expected_scale,
            "{} weight_scale shape must be {:?}, got {:?}",
            self.role.label(),
            expected_scale,
            self.scale_shape
        );
        ensure!(
            self.weight_shape_entries == self.local_experts * 2,
            "{} weight_shape must carry [out_dim, in_dim] for each local expert: expected {} i32 entries, got {}",
            self.role.label(),
            self.local_experts * 2,
            self.weight_shape_entries
        );
        ensure!(
            self.encoding == KimiInt4Encoding::SignedSymmetric,
            "only signed symmetric INT4 is specified for Kimi-K2, got {:?}",
            self.encoding
        );

        Ok(())
    }

    #[must_use]
    pub fn weight_packed_spec(&self) -> TensorSpec {
        self.weight_packed_checkpoint_spec()
    }

    #[must_use]
    pub fn weight_packed_checkpoint_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "u8",
            "expert_major_int4_packed_checkpoint_offset_binary",
            [
                AxisSpec::named("local_expert", self.packed_shape.experts),
                AxisSpec::named("out", self.packed_shape.rows),
                AxisSpec::named("packed_in_over_2", self.packed_shape.cols),
            ],
        )
    }

    #[must_use]
    pub fn weight_packed_cutlass_example69_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "u8",
            "expert_major_int4_packed_cutlass_example69_signed_reordered",
            [
                AxisSpec::named("local_expert", self.packed_shape.experts),
                AxisSpec::named("cutlass_reordered_out", self.packed_shape.rows),
                AxisSpec::named("cutlass_reordered_packed_in_over_2", self.packed_shape.cols),
            ],
        )
    }

    #[must_use]
    pub fn marlin_packed_u32_elements(&self) -> usize {
        self.local_experts * (self.logical_shape.in_dim / 16) * (self.logical_shape.out_dim * 2)
    }

    #[must_use]
    pub fn weight_packed_marlin_uint4b8_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "u32",
            "expert_major_int4_packed_marlin_uint4b8_noact",
            [
                AxisSpec::named("local_expert", self.local_experts),
                AxisSpec::named("in_tile16", self.logical_shape.in_dim / 16),
                AxisSpec::named("out_x2", self.logical_shape.out_dim * 2),
            ],
        )
    }

    #[must_use]
    pub fn weight_scale_spec(&self) -> TensorSpec {
        self.weight_scale_checkpoint_spec()
    }

    #[must_use]
    pub fn weight_scale_checkpoint_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "bf16",
            "expert_major_group_scale_checkpoint",
            [
                AxisSpec::named("local_expert", self.scale_shape.experts),
                AxisSpec::named("out", self.scale_shape.rows),
                AxisSpec::named("in_group", self.scale_shape.cols),
            ],
        )
    }

    #[must_use]
    pub fn weight_scale_cutlass_example69_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "bf16",
            "expert_major_group_scale_cutlass_example69",
            [
                AxisSpec::named("local_expert", self.scale_shape.experts),
                AxisSpec::named("in_group", self.scale_shape.cols),
                AxisSpec::named("out", self.scale_shape.rows),
            ],
        )
    }

    #[must_use]
    pub fn weight_scale_marlin_permuted_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "bf16",
            "expert_major_group_scale_marlin_group_major_perm64",
            [
                AxisSpec::named("local_expert", self.scale_shape.experts),
                AxisSpec::named("in_group", self.scale_shape.cols),
                AxisSpec::named("out", self.scale_shape.rows),
            ],
        )
    }

    #[must_use]
    pub fn weight_shape_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "i32",
            "expert_major_shape",
            [AxisSpec::named("shape_entry", self.weight_shape_entries)],
        )
    }
}

pub struct KimiInt4Weight<'a> {
    pub manifest: KimiInt4WeightManifest,
    pub weight_packed: &'a CudaSlice<u8>,
    pub weight_scale: &'a CudaSlice<bf16>,
    pub weight_shape: &'a CudaSlice<i32>,
}

impl KimiInt4Weight<'_> {
    pub fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        ensure!(
            self.weight_packed.len() == self.manifest.packed_shape.elements(),
            "{} weight_packed len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.packed_shape.elements(),
            self.weight_packed.len()
        );
        ensure!(
            self.weight_scale.len() == self.manifest.scale_shape.elements(),
            "{} weight_scale len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.scale_shape.elements(),
            self.weight_scale.len()
        );
        ensure!(
            self.weight_shape.len() == self.manifest.weight_shape_entries,
            "{} weight_shape len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.weight_shape_entries,
            self.weight_shape.len()
        );
        Ok(())
    }

    #[must_use]
    pub fn dequant_bf16_elements(&self) -> usize {
        self.manifest.local_experts
            * self.manifest.logical_shape.out_dim
            * self.manifest.logical_shape.in_dim
    }
}

pub struct KimiInt4ExpertWeights<'a> {
    pub w1_gate: KimiInt4Weight<'a>,
    pub w3_up: KimiInt4Weight<'a>,
    pub w2_down: KimiInt4Weight<'a>,
}

impl KimiInt4ExpertWeights<'_> {
    pub fn validate(&self) -> Result<()> {
        self.w1_gate.validate()?;
        self.w3_up.validate()?;
        self.w2_down.validate()?;
        ensure_role(&self.w1_gate, KimiInt4ExpertRole::W1Gate)?;
        ensure_role(&self.w3_up, KimiInt4ExpertRole::W3Up)?;
        ensure_role(&self.w2_down, KimiInt4ExpertRole::W2Down)?;

        let offset = self.w1_gate.manifest.local_expert_offset;
        for weight in [&self.w3_up, &self.w2_down] {
            ensure!(
                weight.manifest.local_expert_offset == offset,
                "{} local expert offset must match W1 offset {offset}, got {}",
                weight.manifest.role.label(),
                weight.manifest.local_expert_offset
            );
        }
        Ok(())
    }
}

pub struct KimiMarlinInt4Weight<'a> {
    pub manifest: KimiInt4WeightManifest,
    pub weight_packed_uint4b8: &'a CudaSlice<u8>,
    pub weight_scale_permuted: &'a CudaSlice<bf16>,
}

impl KimiMarlinInt4Weight<'_> {
    pub fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        ensure!(
            self.weight_packed_uint4b8.len() == self.manifest.packed_shape.elements(),
            "{} Marlin uint4b8 packed len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.packed_shape.elements(),
            self.weight_packed_uint4b8.len()
        );
        ensure!(
            self.weight_scale_permuted.len() == self.manifest.scale_shape.elements(),
            "{} Marlin permuted scale len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.scale_shape.elements(),
            self.weight_scale_permuted.len()
        );
        ensure!(
            self.manifest.nibble_order == KimiInt4NibbleOrder::LowThenHigh,
            "{} Marlin package expects low-then-high checkpoint nibbles before repack, got {}",
            self.manifest.role.label(),
            self.manifest.nibble_order.label()
        );
        Ok(())
    }

    #[must_use]
    pub fn manifest_call(&self) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.int4_marlin_weight",
            "Kimi-K2 vLLM Marlin WNA16 INT4 expert weight package",
        )
        .input(
            "weight_packed_uint4b8",
            self.manifest.weight_packed_marlin_uint4b8_spec(),
        )
        .input(
            "weight_scale_permuted",
            self.manifest.weight_scale_marlin_permuted_spec(),
        )
        .attr("encoding", "uint4b8_bias_8".to_string())
        .attr("scale_layout", "vllm_group_major_perm64".to_string())
        .attr("act_order", "false".to_string())
        .attr("group_size", self.manifest.group_size.to_string())
        .attr("local_experts", self.manifest.local_experts.to_string())
    }
}

pub struct KimiMarlinFusedW13Int4Weight<'a> {
    pub local_experts: usize,
    pub in_dim: usize,
    pub intermediate_dim: usize,
    pub group_size: usize,
    pub weight_packed_uint4b8: &'a CudaSlice<u8>,
    pub weight_scale_permuted: &'a CudaSlice<bf16>,
}

impl KimiMarlinFusedW13Int4Weight<'_> {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Marlin fused W13 local_experts must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.local_experts
        );
        ensure!(
            self.in_dim == KIMI_K2_HIDDEN,
            "Marlin fused W13 in_dim must be {}, got {}",
            KIMI_K2_HIDDEN,
            self.in_dim
        );
        ensure!(
            self.intermediate_dim == KIMI_K2_EXPERT_INTERMEDIATE,
            "Marlin fused W13 intermediate_dim must be {}, got {}",
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.intermediate_dim
        );
        ensure!(
            self.group_size == KIMI_K2_INT4_GROUP_SIZE,
            "Marlin fused W13 group_size must be {}, got {}",
            KIMI_K2_INT4_GROUP_SIZE,
            self.group_size
        );
        let expected_packed = self.local_experts * (self.in_dim / 16) * (self.intermediate_dim * 4);
        ensure!(
            self.weight_packed_uint4b8.len() == expected_packed * std::mem::size_of::<u32>(),
            "Marlin fused W13 uint4b8 packed len must be {} bytes, got {}",
            expected_packed * std::mem::size_of::<u32>(),
            self.weight_packed_uint4b8.len()
        );
        let expected_scale =
            self.local_experts * (self.in_dim / self.group_size) * (2 * self.intermediate_dim);
        ensure!(
            self.weight_scale_permuted.len() == expected_scale,
            "Marlin fused W13 permuted scale len must be {}, got {}",
            expected_scale,
            self.weight_scale_permuted.len()
        );
        Ok(())
    }

    #[must_use]
    pub fn manifest_call(&self) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.int4_marlin_w13_weight",
            "Kimi-K2 vLLM Marlin WNA16 fused W13 expert weight package",
        )
        .input(
            "weight_packed_uint4b8",
            TensorSpec::named(
                "u32",
                "expert_major_int4_packed_marlin_w13_uint4b8_noact",
                [
                    AxisSpec::named("local_expert", self.local_experts),
                    AxisSpec::named("in_tile16", self.in_dim / 16),
                    AxisSpec::named("out_x2", 2 * self.intermediate_dim * 2),
                ],
            ),
        )
        .input(
            "weight_scale_permuted",
            TensorSpec::named(
                "bf16",
                "expert_major_group_scale_marlin_w13_group_major_perm64",
                [
                    AxisSpec::named("local_expert", self.local_experts),
                    AxisSpec::named("in_group", self.in_dim / self.group_size),
                    AxisSpec::named("out", 2 * self.intermediate_dim),
                ],
            ),
        )
        .attr("encoding", "uint4b8_bias_8".to_string())
        .attr("scale_layout", "vllm_w13_group_major_perm64".to_string())
        .attr("act_order", "false".to_string())
        .attr("group_size", self.group_size.to_string())
        .attr("w13_order", "gate_then_up".to_string())
    }
}

pub struct KimiMarlinInt4ExpertWeights<'a> {
    pub w13: KimiMarlinFusedW13Int4Weight<'a>,
    pub w2_down: KimiMarlinInt4Weight<'a>,
}

impl KimiMarlinInt4ExpertWeights<'_> {
    pub fn validate(&self) -> Result<()> {
        self.w13.validate()?;
        self.w2_down.validate()?;
        ensure!(
            self.w2_down.manifest.role == KimiInt4ExpertRole::W2Down,
            "Marlin W2 role mismatch: got {:?}",
            self.w2_down.manifest.role
        );
        Ok(())
    }
}

pub struct KimiExpertMajorRoute<'a> {
    pub batch_size: usize,
    pub active_tokens: usize,
    pub routed_tokens: usize,
    pub expert_indptr: &'a CudaSlice<u32>,
}

impl KimiExpertMajorRoute<'_> {
    #[must_use]
    pub const fn max_routed_tokens(active_tokens: usize) -> usize {
        active_tokens * KIMI_K2_TOPK
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be > 0");
        ensure!(self.active_tokens > 0, "active_tokens must be > 0");
        ensure!(
            self.active_tokens >= self.batch_size,
            "active_tokens {} must cover batch_size {} for bs>1 expert-major routing",
            self.active_tokens,
            self.batch_size
        );
        ensure!(
            self.routed_tokens <= Self::max_routed_tokens(self.active_tokens),
            "routed_tokens {} exceeds active_tokens * topk {}",
            self.routed_tokens,
            Self::max_routed_tokens(self.active_tokens)
        );
        ensure!(
            self.expert_indptr.len() == KIMI_K2_LOCAL_EXPERTS + 1,
            "expert_indptr len must be exactly {}, got {}",
            KIMI_K2_LOCAL_EXPERTS + 1,
            self.expert_indptr.len()
        );
        Ok(())
    }

    #[must_use]
    pub fn expert_indptr_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "u32",
            "expert_major",
            [AxisSpec::named(
                "local_expert_plus_one",
                KIMI_K2_LOCAL_EXPERTS + 1,
            )],
        )
    }
}

pub struct KimiExpertMajorRouteWorkspace {
    pub pos_to_token: CudaSlice<i32>,
    pub token_topk_to_pos: CudaSlice<i32>,
    pub expert_indptr: CudaSlice<u32>,
    pub expert_cursor: CudaSlice<u32>,
    pub local_count: CudaSlice<u32>,
    pub max_active_tokens: usize,
    pub topk: usize,
    pub local_experts: usize,
}

impl KimiExpertMajorRouteWorkspace {
    pub fn new(ctx: &DeviceContext, max_active_tokens: usize) -> Result<Self> {
        ensure!(
            max_active_tokens > 0,
            "Kimi expert-major max_active_tokens must be positive"
        );
        let route_capacity = max_active_tokens * KIMI_K2_TOPK;
        Ok(Self {
            pos_to_token: ctx.stream.alloc_zeros(route_capacity)?,
            token_topk_to_pos: ctx.stream.alloc_zeros(route_capacity)?,
            expert_indptr: ctx.stream.alloc_zeros(KIMI_K2_LOCAL_EXPERTS + 1)?,
            expert_cursor: ctx.stream.alloc_zeros(KIMI_K2_LOCAL_EXPERTS)?,
            local_count: ctx.stream.alloc_zeros(1)?,
            max_active_tokens,
            topk: KIMI_K2_TOPK,
            local_experts: KIMI_K2_LOCAL_EXPERTS,
        })
    }

    pub fn validate_for(&self, active_tokens: usize) -> Result<()> {
        ensure!(
            active_tokens > 0,
            "Kimi expert-major active_tokens must be positive"
        );
        ensure!(
            active_tokens <= self.max_active_tokens,
            "active_tokens {} exceeds Kimi expert-major workspace capacity {}",
            active_tokens,
            self.max_active_tokens
        );
        let route_capacity = active_tokens * KIMI_K2_TOPK;
        ensure!(
            self.pos_to_token.len() >= route_capacity,
            "pos_to_token scratch too small: have {}, need {}",
            self.pos_to_token.len(),
            route_capacity
        );
        ensure!(
            self.token_topk_to_pos.len() >= route_capacity,
            "token_topk_to_pos scratch too small: have {}, need {}",
            self.token_topk_to_pos.len(),
            route_capacity
        );
        ensure!(
            self.expert_indptr.len() == KIMI_K2_LOCAL_EXPERTS + 1,
            "expert_indptr len must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS + 1,
            self.expert_indptr.len()
        );
        ensure!(
            self.expert_cursor.len() == KIMI_K2_LOCAL_EXPERTS,
            "expert_cursor len must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.expert_cursor.len()
        );
        ensure!(
            self.local_count.len() == 1,
            "local_count len must be 1, got {}",
            self.local_count.len()
        );
        ensure!(
            self.topk == KIMI_K2_TOPK && self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Kimi expert-major workspace constants must be topk={} local_experts={}",
            KIMI_K2_TOPK,
            KIMI_K2_LOCAL_EXPERTS
        );
        Ok(())
    }

    #[must_use]
    pub const fn route_capacity(active_tokens: usize) -> usize {
        active_tokens * KIMI_K2_TOPK
    }
}

pub struct KimiExpertMajorRouting<'a> {
    pub route: KimiExpertMajorRoute<'a>,
    pub pos_to_token: &'a CudaSlice<i32>,
    pub token_topk_to_pos: &'a CudaSlice<i32>,
    pub local_count: &'a CudaSlice<u32>,
    pub global_expert_start: usize,
}

impl KimiExpertMajorRouting<'_> {
    #[must_use]
    pub fn manifest_call(&self) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.expert_major_route",
            "Kimi-K2 device-resident topk route to expert-major layout",
        )
        .output(
            "expert_indptr",
            TensorSpec::named(
                "u32",
                "expert_major",
                [AxisSpec::named(
                    "local_expert_plus_one",
                    KIMI_K2_LOCAL_EXPERTS + 1,
                )],
            ),
        )
        .output(
            "pos_to_token",
            TensorSpec::named(
                "i32",
                "expert_major",
                [AxisSpec::named("routed_capacity", self.route.routed_tokens)],
            ),
        )
        .output(
            "token_topk_to_pos",
            TensorSpec::named(
                "i32",
                "token_topk",
                [AxisSpec::named(
                    "route_entry",
                    self.route.active_tokens * KIMI_K2_TOPK,
                )],
            ),
        )
        .attr("batch_size", self.route.batch_size.to_string())
        .attr("active_tokens", self.route.active_tokens.to_string())
        .attr("topk", KIMI_K2_TOPK.to_string())
        .attr("local_experts", KIMI_K2_LOCAL_EXPERTS.to_string())
        .attr("global_expert_start", self.global_expert_start.to_string())
        .attr("device_resident_metadata", "true".to_string())
        .attr("decode_step_allocation", "forbidden".to_string())
        .attr("decode_step_d2h", "forbidden".to_string())
    }
}

pub struct KimiMarlinRouteWorkspace {
    pub sorted_token_ids: CudaSlice<i32>,
    pub expert_ids: CudaSlice<i32>,
    pub num_tokens_post_padded: CudaSlice<i32>,
    pub expert_offsets: CudaSlice<u32>,
    pub expert_cursor: CudaSlice<u32>,
    pub max_active_tokens: usize,
    pub max_padded_tokens: usize,
    pub max_m_blocks: usize,
    pub block_size: usize,
    pub topk: usize,
    pub local_experts: usize,
}

impl KimiMarlinRouteWorkspace {
    pub fn new(ctx: &DeviceContext, max_active_tokens: usize, block_size: usize) -> Result<Self> {
        ensure!(
            max_active_tokens > 0,
            "Kimi Marlin route max_active_tokens must be positive"
        );
        validate_marlin_block_size(block_size)?;
        let max_padded_tokens = marlin_padded_route_capacity(max_active_tokens, block_size)?;
        let max_m_blocks = max_padded_tokens.div_ceil(block_size);
        Ok(Self {
            sorted_token_ids: ctx.stream.alloc_zeros(max_padded_tokens)?,
            expert_ids: ctx.stream.alloc_zeros(max_m_blocks)?,
            num_tokens_post_padded: ctx.stream.alloc_zeros(1)?,
            expert_offsets: ctx.stream.alloc_zeros(KIMI_K2_LOCAL_EXPERTS + 1)?,
            expert_cursor: ctx.stream.alloc_zeros(KIMI_K2_LOCAL_EXPERTS)?,
            max_active_tokens,
            max_padded_tokens,
            max_m_blocks,
            block_size,
            topk: KIMI_K2_TOPK,
            local_experts: KIMI_K2_LOCAL_EXPERTS,
        })
    }

    pub fn validate_for(&self, active_tokens: usize) -> Result<()> {
        validate_marlin_block_size(self.block_size)?;
        ensure!(
            active_tokens > 0,
            "Kimi Marlin route active_tokens must be positive"
        );
        ensure!(
            active_tokens <= self.max_active_tokens,
            "active_tokens {} exceeds Kimi Marlin route workspace capacity {}",
            active_tokens,
            self.max_active_tokens
        );
        let required_padded = marlin_padded_route_capacity(active_tokens, self.block_size)?;
        let required_blocks = required_padded.div_ceil(self.block_size);
        ensure!(
            self.max_padded_tokens >= required_padded
                && self.sorted_token_ids.len() >= self.max_padded_tokens,
            "Marlin sorted_token_ids capacity too small: have {} metadata/{} slice, need {}",
            self.max_padded_tokens,
            self.sorted_token_ids.len(),
            required_padded
        );
        ensure!(
            self.max_m_blocks >= required_blocks && self.expert_ids.len() >= self.max_m_blocks,
            "Marlin expert_ids capacity too small: have {} metadata/{} slice, need {}",
            self.max_m_blocks,
            self.expert_ids.len(),
            required_blocks
        );
        ensure!(
            self.num_tokens_post_padded.len() == 1,
            "num_tokens_post_padded len must be 1, got {}",
            self.num_tokens_post_padded.len()
        );
        ensure!(
            self.expert_offsets.len() == KIMI_K2_LOCAL_EXPERTS + 1,
            "expert_offsets len must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS + 1,
            self.expert_offsets.len()
        );
        ensure!(
            self.expert_cursor.len() == KIMI_K2_LOCAL_EXPERTS,
            "expert_cursor len must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.expert_cursor.len()
        );
        ensure!(
            self.topk == KIMI_K2_TOPK && self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Kimi Marlin route workspace constants must be topk={} local_experts={}",
            KIMI_K2_TOPK,
            KIMI_K2_LOCAL_EXPERTS
        );
        Ok(())
    }
}

pub struct KimiMarlinWna16Workspace {
    pub locks: CudaSlice<i32>,
    pub c_tmp: CudaSlice<f32>,
    pub max_m_blocks: usize,
    pub max_padded_tokens: usize,
    pub max_size_n: usize,
    pub block_size: usize,
}

impl KimiMarlinWna16Workspace {
    pub fn new(
        ctx: &DeviceContext,
        max_m_blocks: usize,
        max_size_n: usize,
        block_size: usize,
    ) -> Result<Self> {
        validate_marlin_block_size(block_size)?;
        ensure!(
            max_m_blocks > 0,
            "Kimi Marlin WNA16 max_m_blocks must be > 0"
        );
        ensure!(
            max_size_n >= KIMI_K2_EXPERT_INTERMEDIATE && max_size_n % 64 == 0,
            "Kimi Marlin WNA16 max_size_n must be >= {} and divisible by 64, got {}",
            KIMI_K2_EXPERT_INTERMEDIATE,
            max_size_n
        );
        let lock_count = (max_size_n / 64)
            .checked_mul(max_m_blocks)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 workspace size overflow"))?
            .max(1);
        let max_padded_tokens = max_m_blocks
            .checked_mul(block_size)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 padded token capacity overflow"))?;
        let mut c_tmp_elements = max_size_n
            .checked_mul(max_padded_tokens)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp capacity overflow"))?;
        if block_size == 8 {
            c_tmp_elements = c_tmp_elements
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp capacity overflow"))?;
        }
        Ok(Self {
            locks: ctx.stream.alloc_zeros(lock_count)?,
            c_tmp: ctx.stream.alloc_zeros(c_tmp_elements.max(1))?,
            max_m_blocks,
            max_padded_tokens,
            max_size_n,
            block_size,
        })
    }

    pub fn validate_for(&self, routing: &KimiMarlinRouting<'_>, size_n: usize) -> Result<()> {
        validate_marlin_block_size(self.block_size)?;
        ensure!(
            self.block_size == routing.block_size,
            "Kimi Marlin WNA16 workspace block_size {} must match routing {}",
            self.block_size,
            routing.block_size
        );
        ensure!(
            routing.max_m_blocks <= self.max_m_blocks,
            "Kimi Marlin WNA16 workspace max_m_blocks {} below routing {}",
            self.max_m_blocks,
            routing.max_m_blocks
        );
        ensure!(
            routing.max_padded_tokens <= self.max_padded_tokens,
            "Kimi Marlin WNA16 workspace max_padded_tokens {} below routing {}",
            self.max_padded_tokens,
            routing.max_padded_tokens
        );
        ensure!(
            size_n <= self.max_size_n && size_n % 64 == 0,
            "Kimi Marlin WNA16 size_n {} exceeds workspace max {} or is not divisible by 64",
            size_n,
            self.max_size_n
        );
        let required = (size_n / 64)
            .checked_mul(routing.max_m_blocks)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 required workspace overflow"))?
            .max(1);
        ensure!(
            self.locks.len() >= required,
            "Kimi Marlin WNA16 workspace lock len must cover {}, got {}",
            required,
            self.locks.len()
        );
        let mut required_c_tmp = size_n
            .checked_mul(routing.max_padded_tokens)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp required overflow"))?;
        if self.block_size == 8 {
            required_c_tmp = required_c_tmp
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp required overflow"))?;
        }
        ensure!(
            self.c_tmp.len() >= required_c_tmp.max(1),
            "Kimi Marlin WNA16 c_tmp len must cover {}, got {}",
            required_c_tmp,
            self.c_tmp.len()
        );
        Ok(())
    }
}

pub struct KimiMarlinRouting<'a> {
    pub batch_size: usize,
    pub active_tokens: usize,
    pub route_elems: usize,
    pub global_expert_start: usize,
    pub block_size: usize,
    pub max_padded_tokens: usize,
    pub max_m_blocks: usize,
    pub sorted_token_ids: &'a CudaSlice<i32>,
    pub expert_ids: &'a CudaSlice<i32>,
    pub num_tokens_post_padded: &'a CudaSlice<i32>,
}

impl KimiMarlinRouting<'_> {
    #[must_use]
    pub fn manifest_call(&self) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.marlin_align_block_size",
            "Kimi-K2 vLLM Marlin WNA16 route alignment metadata",
        )
        .output(
            "sorted_token_ids",
            TensorSpec::named(
                "i32",
                "marlin_sorted_route_ids_padded",
                [AxisSpec::named("max_padded_tokens", self.max_padded_tokens)],
            ),
        )
        .output(
            "expert_ids",
            TensorSpec::named(
                "i32",
                "marlin_expert_id_per_m_block",
                [AxisSpec::named("max_m_blocks", self.max_m_blocks)],
            ),
        )
        .output(
            "num_tokens_post_padded",
            TensorSpec::named("i32", "scalar_device", [AxisSpec::named("value", 1)]),
        )
        .attr("batch_size", self.batch_size.to_string())
        .attr("active_tokens", self.active_tokens.to_string())
        .attr("route_elems", self.route_elems.to_string())
        .attr("topk", KIMI_K2_TOPK.to_string())
        .attr("local_experts", KIMI_K2_LOCAL_EXPERTS.to_string())
        .attr("global_expert_start", self.global_expert_start.to_string())
        .attr("block_size", self.block_size.to_string())
        .attr("sentinel_token_id", self.route_elems.to_string())
        .attr("device_resident_metadata", "true".to_string())
        .attr("decode_step_allocation", "forbidden".to_string())
        .attr("decode_step_d2h", "forbidden".to_string())
    }
}

pub struct KimiGroupedW1W3Plan<'a> {
    pub route: KimiExpertMajorRoute<'a>,
    pub expert_hidden: &'a HiddenStates,
    pub gate_out: &'a mut HiddenStates,
    pub up_out: &'a mut HiddenStates,
}

impl KimiGroupedW1W3Plan<'_> {
    pub fn validate(&self, weights: &KimiInt4ExpertWeights<'_>) -> Result<()> {
        weights.validate()?;
        self.route.validate()?;
        validate_hidden_states(
            "w1w3.expert_hidden",
            self.expert_hidden,
            KIMI_K2_HIDDEN,
            self.route.routed_tokens,
        )?;
        validate_hidden_states(
            "w1w3.gate_out",
            self.gate_out,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )?;
        validate_hidden_states(
            "w1w3.up_out",
            self.up_out,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )
    }

    #[must_use]
    pub fn manifest_call(&self, weights: &KimiInt4ExpertWeights<'_>) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.int4_grouped_w1_w3",
            "Kimi-K2 INT4 grouped W1/W3 expert projection",
        )
        .input(
            "expert_hidden",
            hidden_spec(KIMI_K2_HIDDEN, self.route.routed_tokens),
        )
        .input(
            "w1_weight_packed",
            weights
                .w1_gate
                .manifest
                .weight_packed_cutlass_example69_spec(),
        )
        .input(
            "w1_weight_scale",
            weights
                .w1_gate
                .manifest
                .weight_scale_cutlass_example69_spec(),
        )
        .input(
            "w1_weight_shape",
            weights.w1_gate.manifest.weight_shape_spec(),
        )
        .input(
            "w3_weight_packed",
            weights
                .w3_up
                .manifest
                .weight_packed_cutlass_example69_spec(),
        )
        .input(
            "w3_weight_scale",
            weights.w3_up.manifest.weight_scale_cutlass_example69_spec(),
        )
        .input(
            "w3_weight_shape",
            weights.w3_up.manifest.weight_shape_spec(),
        )
        .input("expert_indptr", self.route.expert_indptr_spec())
        .output(
            "gate_out",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .output(
            "up_out",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .attr("local_experts", KIMI_K2_LOCAL_EXPERTS.to_string())
        .attr("topk", KIMI_K2_TOPK.to_string())
        .attr("group_size", KIMI_K2_INT4_GROUP_SIZE.to_string())
        .attr("batch_size", self.route.batch_size.to_string())
        .attr("active_tokens", self.route.active_tokens.to_string())
        .attr("routed_tokens", self.route.routed_tokens.to_string())
        .attr(
            "expert_indptr_entries",
            (KIMI_K2_LOCAL_EXPERTS + 1).to_string(),
        )
        .attr("layout", "expert_major_routed_tokens".to_string())
        .attr(
            "weight_encoding",
            weights.w1_gate.manifest.encoding.label().to_string(),
        )
        .attr(
            "nibble_order",
            weights.w1_gate.manifest.nibble_order.label().to_string(),
        )
        .attr("scale_dtype", "bf16".to_string())
        .attr("accumulator_dtype", "f32".to_string())
        .attr(
            "backend",
            "cutlass_cpp_aot_sm90a_example69_limitation_probe".to_string(),
        )
        .attr(
            "correctness_backend",
            "false_per32_scale_semantics_not_supported".to_string(),
        )
        .attr(
            "w1_w3_fused_n",
            (2 * KIMI_K2_EXPERT_INTERMEDIATE).to_string(),
        )
        .attr(
            "cuda_body",
            "cutlass_grouped_projection_workspace".to_string(),
        )
        .attr(
            "cuda_graph_ready",
            "requires_prepared_device_resident_cutlass_workspace".to_string(),
        )
    }
}

pub struct KimiGroupedW2SwiGluPlan<'a> {
    pub route: KimiExpertMajorRoute<'a>,
    pub gate: &'a HiddenStates,
    pub up: &'a HiddenStates,
    pub expert_output: &'a mut HiddenStates,
}

impl KimiGroupedW2SwiGluPlan<'_> {
    pub fn validate(&self, weights: &KimiInt4ExpertWeights<'_>) -> Result<()> {
        weights.validate()?;
        self.route.validate()?;
        validate_hidden_states(
            "w2_swiglu.gate",
            self.gate,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )?;
        validate_hidden_states(
            "w2_swiglu.up",
            self.up,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )?;
        validate_hidden_states(
            "w2_swiglu.expert_output",
            self.expert_output,
            KIMI_K2_HIDDEN,
            self.route.routed_tokens,
        )
    }

    #[must_use]
    pub fn manifest_call(&self, weights: &KimiInt4ExpertWeights<'_>) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.int4_grouped_w2_swiglu",
            "Kimi-K2 INT4 grouped W2 SwiGLU expert projection",
        )
        .input(
            "gate",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .input(
            "up",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .input(
            "w2_weight_packed",
            weights
                .w2_down
                .manifest
                .weight_packed_cutlass_example69_spec(),
        )
        .input(
            "w2_weight_scale",
            weights
                .w2_down
                .manifest
                .weight_scale_cutlass_example69_spec(),
        )
        .input(
            "w2_weight_shape",
            weights.w2_down.manifest.weight_shape_spec(),
        )
        .input("expert_indptr", self.route.expert_indptr_spec())
        .output(
            "expert_output",
            hidden_spec(KIMI_K2_HIDDEN, self.route.routed_tokens),
        )
        .attr("local_experts", KIMI_K2_LOCAL_EXPERTS.to_string())
        .attr("topk", KIMI_K2_TOPK.to_string())
        .attr("group_size", KIMI_K2_INT4_GROUP_SIZE.to_string())
        .attr("batch_size", self.route.batch_size.to_string())
        .attr("active_tokens", self.route.active_tokens.to_string())
        .attr("routed_tokens", self.route.routed_tokens.to_string())
        .attr(
            "expert_indptr_entries",
            (KIMI_K2_LOCAL_EXPERTS + 1).to_string(),
        )
        .attr("layout", "expert_major_routed_tokens".to_string())
        .attr(
            "weight_encoding",
            weights.w2_down.manifest.encoding.label().to_string(),
        )
        .attr(
            "nibble_order",
            weights.w2_down.manifest.nibble_order.label().to_string(),
        )
        .attr("scale_dtype", "bf16".to_string())
        .attr("activation", "external_silu_gate_mul_up".to_string())
        .attr("accumulator_dtype", "f32".to_string())
        .attr(
            "backend",
            "cutlass_cpp_aot_sm90a_example69_limitation_probe".to_string(),
        )
        .attr(
            "correctness_backend",
            "false_per32_scale_semantics_not_supported".to_string(),
        )
        .attr(
            "cuda_body",
            "cutlass_grouped_projection_workspace".to_string(),
        )
        .attr(
            "cuda_graph_ready",
            "requires_prepared_device_resident_cutlass_workspace".to_string(),
        )
    }
}

pub fn kimi_int4_metadata_probe(ctx: &DeviceContext, weight: &KimiInt4Weight<'_>) -> Result<()> {
    weight.validate()?;
    let (shape_ptr, _shape_guard) = weight.weight_shape.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::kimi_int4_expert_metadata_probe_cuda(
            shape_ptr as *const i32,
            weight.manifest.weight_shape_entries,
            weight.manifest.local_experts as i32,
            weight.manifest.logical_shape.in_dim as i32,
            weight.manifest.logical_shape.out_dim as i32,
            weight.manifest.group_size as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_cutlass_int4_sm90a_support() -> Result<()> {
    let result = unsafe { ffi::kimi_cutlass_int4_sm90a_support_cuda() };
    result.result()?;
    Ok(())
}

pub fn kimi_cutlass_int4_sm90a_support_probe() -> Result<KimiCutlassSm90aSupport> {
    let mut supported = 0;
    let mut sm_major = 0;
    let mut sm_minor = 0;
    let result = unsafe {
        ffi::kimi_cutlass_int4_sm90a_support_probe_cuda(
            &raw mut supported,
            &raw mut sm_major,
            &raw mut sm_minor,
        )
    };
    result.result()?;
    Ok(KimiCutlassSm90aSupport {
        supported: supported != 0,
        sm_major,
        sm_minor,
    })
}

pub fn kimi_cutlass_int4_grouped_workspace_sizes(
    max_routed_tokens: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<ffi::KimiCutlassInt4GroupedWorkspaceSizes> {
    ensure!(
        max_routed_tokens <= i32::MAX as usize,
        "max_routed_tokens {} exceeds i32::MAX",
        max_routed_tokens
    );
    ensure!(
        in_dim <= i32::MAX as usize,
        "in_dim {in_dim} exceeds i32::MAX"
    );
    ensure!(
        out_dim <= i32::MAX as usize,
        "out_dim {out_dim} exceeds i32::MAX"
    );
    let mut sizes = ffi::KimiCutlassInt4GroupedWorkspaceSizes::default();
    let result = unsafe {
        ffi::kimi_cutlass_int4_grouped_workspace_sizes_sm90a_cuda(
            max_routed_tokens as i32,
            in_dim as i32,
            out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            &raw mut sizes,
        )
    };
    result.result()?;
    Ok(sizes)
}

pub fn kimi_cutlass_int4_grouped_prepare(
    ctx: &DeviceContext,
    workspace: &mut KimiCutlassInt4GroupedWorkspace,
    route: &KimiExpertMajorRoute<'_>,
    input: &HiddenStates,
    weight: &KimiInt4Weight<'_>,
    output: &mut HiddenStates,
) -> Result<()> {
    validate_cutlass_projection("cutlass_prepare", workspace, route, input, weight, output)?;
    let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.weight_packed.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = weight.weight_scale.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = route.expert_indptr.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);
    let (workspace_ptr, _workspace_guard) = workspace.storage.device_ptr_mut(&ctx.stream);

    let params = ffi::KimiCutlassInt4GroupedLaunchParams {
        input: input_ptr as *const ffi::Half,
        weight_packed_reordered: weight_ptr as *const u8,
        weight_scale: scale_ptr as *const ffi::Half,
        expert_indptr: expert_ptr as *const u32,
        output: output_ptr as *mut ffi::Half,
        workspace: workspace_ptr as *mut std::ffi::c_void,
        workspace_bytes: workspace.sizes.total_bytes,
        routed_tokens: route.routed_tokens as i32,
        in_dim: weight.manifest.logical_shape.in_dim as i32,
        out_dim: weight.manifest.logical_shape.out_dim as i32,
        local_experts: KIMI_K2_LOCAL_EXPERTS as i32,
        group_size: KIMI_K2_INT4_GROUP_SIZE as i32,
        sm_count: 0,
    };
    let result = unsafe {
        ffi::kimi_cutlass_int4_grouped_prepare_sm90a_cuda(params, ctx.stream.cu_stream())
    };
    result.result()?;
    Ok(())
}

pub fn kimi_cutlass_int4_grouped_launch(
    ctx: &DeviceContext,
    workspace: &mut KimiCutlassInt4GroupedWorkspace,
    route: &KimiExpertMajorRoute<'_>,
    input: &HiddenStates,
    weight: &KimiInt4Weight<'_>,
    output: &mut HiddenStates,
) -> Result<()> {
    validate_cutlass_projection("cutlass_launch", workspace, route, input, weight, output)?;
    let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.weight_packed.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = weight.weight_scale.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = route.expert_indptr.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);
    let (workspace_ptr, _workspace_guard) = workspace.storage.device_ptr_mut(&ctx.stream);

    let params = ffi::KimiCutlassInt4GroupedLaunchParams {
        input: input_ptr as *const ffi::Half,
        weight_packed_reordered: weight_ptr as *const u8,
        weight_scale: scale_ptr as *const ffi::Half,
        expert_indptr: expert_ptr as *const u32,
        output: output_ptr as *mut ffi::Half,
        workspace: workspace_ptr as *mut std::ffi::c_void,
        workspace_bytes: workspace.sizes.total_bytes,
        routed_tokens: route.routed_tokens as i32,
        in_dim: weight.manifest.logical_shape.in_dim as i32,
        out_dim: weight.manifest.logical_shape.out_dim as i32,
        local_experts: KIMI_K2_LOCAL_EXPERTS as i32,
        group_size: KIMI_K2_INT4_GROUP_SIZE as i32,
        sm_count: 0,
    };
    let result =
        unsafe { ffi::kimi_cutlass_int4_grouped_launch_sm90a_cuda(params, ctx.stream.cu_stream()) };
    result.result()?;
    Ok(())
}

pub fn kimi_moe_marlin_align_block_size<'a>(
    ctx: &DeviceContext,
    workspace: &'a mut KimiMarlinRouteWorkspace,
    topk_idx: &CudaSlice<i32>,
    batch_size: usize,
    active_tokens: usize,
    global_expert_start: usize,
) -> Result<KimiMarlinRouting<'a>> {
    workspace.validate_for(active_tokens)?;
    validate_global_expert_start(global_expert_start)?;
    ensure!(batch_size > 0, "batch_size must be > 0");
    ensure!(
        active_tokens >= batch_size,
        "active_tokens {} must cover batch_size {} for bs>1 Marlin routing",
        active_tokens,
        batch_size
    );
    let route_elems = active_tokens
        .checked_mul(KIMI_K2_TOPK)
        .ok_or_else(|| anyhow::anyhow!("active_tokens * topk overflow"))?;
    ensure!(
        route_elems <= i32::MAX as usize,
        "route_elems {route_elems} exceeds i32::MAX"
    );
    ensure!(
        topk_idx.len() >= route_elems,
        "topk_idx len must cover active_tokens * topk: have {}, need {}",
        topk_idx.len(),
        route_elems
    );
    ensure!(
        workspace.max_padded_tokens <= i32::MAX as usize,
        "max_padded_tokens {} exceeds i32::MAX",
        workspace.max_padded_tokens
    );
    ensure!(
        workspace.max_m_blocks <= i32::MAX as usize,
        "max_m_blocks {} exceeds i32::MAX",
        workspace.max_m_blocks
    );

    {
        let (topk_ptr, _topk_guard) = topk_idx.device_ptr(&ctx.stream);
        let (sorted_ptr, _sorted_guard) = workspace.sorted_token_ids.device_ptr_mut(&ctx.stream);
        let (expert_ids_ptr, _expert_ids_guard) = workspace.expert_ids.device_ptr_mut(&ctx.stream);
        let (num_tokens_ptr, _num_tokens_guard) =
            workspace.num_tokens_post_padded.device_ptr_mut(&ctx.stream);
        let (offsets_ptr, _offsets_guard) = workspace.expert_offsets.device_ptr_mut(&ctx.stream);
        let (cursor_ptr, _cursor_guard) = workspace.expert_cursor.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::kimi_moe_marlin_align_block_size_cuda(
                topk_ptr as *const i32,
                sorted_ptr as *mut i32,
                expert_ids_ptr as *mut i32,
                num_tokens_ptr as *mut i32,
                offsets_ptr as *mut u32,
                cursor_ptr as *mut u32,
                active_tokens as i32,
                KIMI_K2_TOPK as i32,
                global_expert_start as i32,
                KIMI_K2_LOCAL_EXPERTS as i32,
                workspace.block_size as i32,
                workspace.max_padded_tokens as i32,
                workspace.max_m_blocks as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }
    Ok(KimiMarlinRouting {
        batch_size,
        active_tokens,
        route_elems,
        global_expert_start,
        block_size: workspace.block_size,
        max_padded_tokens: workspace.max_padded_tokens,
        max_m_blocks: workspace.max_m_blocks,
        sorted_token_ids: &workspace.sorted_token_ids,
        expert_ids: &workspace.expert_ids,
        num_tokens_post_padded: &workspace.num_tokens_post_padded,
    })
}

pub fn kimi_cutlass_int4_reorder_weight(
    ctx: &DeviceContext,
    weight_packed_offset_binary: &CudaSlice<u8>,
    weight_packed_reordered: &mut CudaSlice<u8>,
    manifest: &KimiInt4WeightManifest,
) -> Result<()> {
    manifest.validate()?;
    let expected_bytes = manifest.packed_shape.elements();
    ensure!(
        weight_packed_offset_binary.len() == expected_bytes,
        "{} offset-binary packed len must be {}, got {}",
        manifest.role.label(),
        expected_bytes,
        weight_packed_offset_binary.len()
    );
    ensure!(
        weight_packed_reordered.len() == expected_bytes,
        "{} reordered packed len must be {}, got {}",
        manifest.role.label(),
        expected_bytes,
        weight_packed_reordered.len()
    );
    ensure!(
        manifest.nibble_order == KimiInt4NibbleOrder::LowThenHigh,
        "{} reorder currently expects low-then-high offset-binary INT4, got {}",
        manifest.role.label(),
        manifest.nibble_order.label()
    );
    let (src_ptr, _src_guard) = weight_packed_offset_binary.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = weight_packed_reordered.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_cutlass_int4_reorder_weight_sm90a_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            manifest.logical_shape.in_dim as i32,
            manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_cutlass_int4_reorder_scale(
    ctx: &DeviceContext,
    weight_scale_checkpoint: &CudaSlice<bf16>,
    weight_scale_reordered: &mut CudaSlice<bf16>,
    manifest: &KimiInt4WeightManifest,
) -> Result<()> {
    manifest.validate()?;
    let expected_elements = manifest.scale_shape.elements();
    ensure!(
        weight_scale_checkpoint.len() == expected_elements,
        "{} checkpoint scale len must be {}, got {}",
        manifest.role.label(),
        expected_elements,
        weight_scale_checkpoint.len()
    );
    ensure!(
        weight_scale_reordered.len() == expected_elements,
        "{} reordered scale len must be {}, got {}",
        manifest.role.label(),
        expected_elements,
        weight_scale_reordered.len()
    );
    let (src_ptr, _src_guard) = weight_scale_checkpoint.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = weight_scale_reordered.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_cutlass_int4_reorder_scale_sm90a_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            manifest.logical_shape.in_dim as i32,
            manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_int4_reorder_scale(
    ctx: &DeviceContext,
    weight_scale_checkpoint: &CudaSlice<bf16>,
    weight_scale_marlin: &mut CudaSlice<bf16>,
    manifest: &KimiInt4WeightManifest,
) -> Result<()> {
    manifest.validate()?;
    let expected_elements = manifest.scale_shape.elements();
    ensure!(
        weight_scale_checkpoint.len() == expected_elements,
        "{} checkpoint scale len must be {}, got {}",
        manifest.role.label(),
        expected_elements,
        weight_scale_checkpoint.len()
    );
    ensure!(
        weight_scale_marlin.len() == expected_elements,
        "{} Marlin scale len must be {}, got {}",
        manifest.role.label(),
        expected_elements,
        weight_scale_marlin.len()
    );
    ensure!(
        expected_elements / KIMI_K2_LOCAL_EXPERTS % 64 == 0,
        "{} Marlin scale elements per expert must be divisible by 64, got {}",
        manifest.role.label(),
        expected_elements / KIMI_K2_LOCAL_EXPERTS
    );
    let (src_ptr, _src_guard) = weight_scale_checkpoint.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = weight_scale_marlin.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_int4_reorder_scale_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            manifest.logical_shape.in_dim as i32,
            manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_int4_reorder_weight(
    ctx: &DeviceContext,
    weight_packed_checkpoint_offset_binary: &CudaSlice<u8>,
    weight_packed_marlin: &mut CudaSlice<u8>,
    manifest: &KimiInt4WeightManifest,
) -> Result<()> {
    manifest.validate()?;
    let expected_bytes = manifest.packed_shape.elements();
    ensure!(
        weight_packed_checkpoint_offset_binary.len() == expected_bytes,
        "{} checkpoint packed len must be {}, got {}",
        manifest.role.label(),
        expected_bytes,
        weight_packed_checkpoint_offset_binary.len()
    );
    ensure!(
        weight_packed_marlin.len() == expected_bytes,
        "{} Marlin packed len must be {}, got {}",
        manifest.role.label(),
        expected_bytes,
        weight_packed_marlin.len()
    );
    ensure!(
        manifest.nibble_order == KimiInt4NibbleOrder::LowThenHigh,
        "{} Marlin repack expects low-then-high offset-binary INT4, got {}",
        manifest.role.label(),
        manifest.nibble_order.label()
    );
    ensure!(
        manifest.marlin_packed_u32_elements() * std::mem::size_of::<u32>() == expected_bytes,
        "{} Marlin packed u32 view must preserve checkpoint byte size",
        manifest.role.label()
    );
    ensure!(
        manifest.logical_shape.in_dim.is_multiple_of(16)
            && manifest.logical_shape.out_dim.is_multiple_of(64),
        "{} Marlin repack requires in_dim multiple of 16 and out_dim multiple of 64, got in={} out={}",
        manifest.role.label(),
        manifest.logical_shape.in_dim,
        manifest.logical_shape.out_dim
    );
    let (src_ptr, _src_guard) = weight_packed_checkpoint_offset_binary.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = weight_packed_marlin.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_int4_reorder_weight_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            manifest.logical_shape.in_dim as i32,
            manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_int4_fuse_w13(
    ctx: &DeviceContext,
    gate: &KimiMarlinInt4Weight<'_>,
    up: &KimiMarlinInt4Weight<'_>,
    weight_packed_w13: &mut CudaSlice<u8>,
    weight_scale_w13: &mut CudaSlice<bf16>,
) -> Result<()> {
    gate.validate()?;
    up.validate()?;
    ensure!(
        gate.manifest.role == KimiInt4ExpertRole::W1Gate,
        "Marlin W13 fuse gate role must be W1Gate, got {:?}",
        gate.manifest.role
    );
    ensure!(
        up.manifest.role == KimiInt4ExpertRole::W3Up,
        "Marlin W13 fuse up role must be W3Up, got {:?}",
        up.manifest.role
    );
    ensure!(
        gate.manifest.local_expert_offset == up.manifest.local_expert_offset,
        "Marlin W13 fuse requires matching expert ranges, got {} and {}",
        gate.manifest.local_expert_offset,
        up.manifest.local_expert_offset
    );
    ensure!(
        gate.manifest.logical_shape == up.manifest.logical_shape,
        "Marlin W13 fuse requires matching shapes, got {:?} and {:?}",
        gate.manifest.logical_shape,
        up.manifest.logical_shape
    );
    let expected_weight_len = gate.weight_packed_uint4b8.len() + up.weight_packed_uint4b8.len();
    ensure!(
        weight_packed_w13.len() == expected_weight_len,
        "Marlin fused W13 packed len must be {}, got {}",
        expected_weight_len,
        weight_packed_w13.len()
    );
    let expected_scale_len = gate.weight_scale_permuted.len() + up.weight_scale_permuted.len();
    ensure!(
        weight_scale_w13.len() == expected_scale_len,
        "Marlin fused W13 scale len must be {}, got {}",
        expected_scale_len,
        weight_scale_w13.len()
    );

    let (gate_weight_ptr, _gate_weight_guard) = gate.weight_packed_uint4b8.device_ptr(&ctx.stream);
    let (up_weight_ptr, _up_weight_guard) = up.weight_packed_uint4b8.device_ptr(&ctx.stream);
    let (w13_weight_ptr, _w13_weight_guard) = weight_packed_w13.device_ptr_mut(&ctx.stream);
    let (gate_scale_ptr, _gate_scale_guard) = gate.weight_scale_permuted.device_ptr(&ctx.stream);
    let (up_scale_ptr, _up_scale_guard) = up.weight_scale_permuted.device_ptr(&ctx.stream);
    let (w13_scale_ptr, _w13_scale_guard) = weight_scale_w13.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_int4_fuse_w13_cuda(
            gate_weight_ptr as *const u8,
            up_weight_ptr as *const u8,
            w13_weight_ptr as *mut u8,
            gate_scale_ptr as *const ffi::Half,
            up_scale_ptr as *const ffi::Half,
            w13_scale_ptr as *mut ffi::Half,
            gate.manifest.logical_shape.in_dim as i32,
            gate.manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_wna16_w13_gemm(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &HiddenStates,
    weight: &KimiMarlinFusedW13Int4Weight<'_>,
    topk_weight: &CudaSlice<f32>,
    output_w13: &mut HiddenStates,
) -> Result<()> {
    weight.validate()?;
    workspace.validate_for(routing, 2 * KIMI_K2_EXPERT_INTERMEDIATE)?;
    validate_hidden_states(
        "marlin_w13.input",
        input,
        KIMI_K2_HIDDEN,
        routing.active_tokens,
    )?;
    validate_hidden_states(
        "marlin_w13.output",
        output_w13,
        2 * KIMI_K2_EXPERT_INTERMEDIATE,
        routing.route_elems,
    )?;
    ensure!(
        topk_weight.len() >= routing.route_elems,
        "topk_weight len must cover {}, got {}",
        routing.route_elems,
        topk_weight.len()
    );
    launch_marlin_wna16_gemm(
        ctx,
        workspace,
        routing,
        input,
        weight.weight_packed_uint4b8,
        weight.weight_scale_permuted,
        topk_weight,
        output_w13,
        KIMI_K2_TOPK,
        false,
        routing.active_tokens,
        2 * KIMI_K2_EXPERT_INTERMEDIATE,
        KIMI_K2_HIDDEN,
    )
}

pub fn kimi_marlin_wna16_w2_gemm(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &HiddenStates,
    weight: &KimiMarlinInt4Weight<'_>,
    topk_weight: &CudaSlice<f32>,
    output: &mut HiddenStates,
) -> Result<()> {
    weight.validate()?;
    ensure!(
        weight.manifest.role == KimiInt4ExpertRole::W2Down,
        "Marlin W2 role mismatch: got {:?}",
        weight.manifest.role
    );
    workspace.validate_for(routing, KIMI_K2_HIDDEN)?;
    validate_hidden_states(
        "marlin_w2.input",
        input,
        KIMI_K2_EXPERT_INTERMEDIATE,
        routing.route_elems,
    )?;
    validate_hidden_states(
        "marlin_w2.output",
        output,
        KIMI_K2_HIDDEN,
        routing.route_elems,
    )?;
    ensure!(
        topk_weight.len() >= routing.route_elems,
        "topk_weight len must cover {}, got {}",
        routing.route_elems,
        topk_weight.len()
    );
    launch_marlin_wna16_gemm(
        ctx,
        workspace,
        routing,
        input,
        weight.weight_packed_uint4b8,
        weight.weight_scale_permuted,
        topk_weight,
        output,
        1,
        true,
        routing.route_elems,
        KIMI_K2_HIDDEN,
        KIMI_K2_EXPERT_INTERMEDIATE,
    )
}

pub fn kimi_marlin_w13_swiglu(
    ctx: &DeviceContext,
    w13: &HiddenStates,
    output: &mut HiddenStates,
) -> Result<()> {
    validate_hidden_states(
        "marlin_w13_swiglu.input",
        w13,
        2 * KIMI_K2_EXPERT_INTERMEDIATE,
        output.seq_len,
    )?;
    validate_hidden_states(
        "marlin_w13_swiglu.output",
        output,
        KIMI_K2_EXPERT_INTERMEDIATE,
        w13.seq_len,
    )?;
    let (w13_ptr, _w13_guard) = w13.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_w13_swiglu_cuda(
            w13_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            w13.seq_len as i32,
            KIMI_K2_EXPERT_INTERMEDIATE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_sum_topk_rows_f32(
    ctx: &DeviceContext,
    route_output: &HiddenStates,
    active_tokens: usize,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    validate_hidden_states(
        "marlin_sum_topk.route_output",
        route_output,
        KIMI_K2_HIDDEN,
        active_tokens * KIMI_K2_TOPK,
    )?;
    ensure!(
        out.len() >= active_tokens * KIMI_K2_HIDDEN,
        "marlin_sum_topk output too small: have {}, need {}",
        out.len(),
        active_tokens * KIMI_K2_HIDDEN
    );
    let (route_ptr, _route_guard) = route_output.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_sum_topk_rows_f32_cuda(
            route_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            active_tokens as i32,
            KIMI_K2_TOPK as i32,
            KIMI_K2_HIDDEN as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn launch_marlin_wna16_gemm(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &HiddenStates,
    weight_packed_uint4b8: &CudaSlice<u8>,
    weight_scale_permuted: &CudaSlice<bf16>,
    topk_weight: &CudaSlice<f32>,
    output: &mut HiddenStates,
    top_k: usize,
    mul_topk_weights: bool,
    size_m: usize,
    size_n: usize,
    size_k: usize,
) -> Result<()> {
    ensure!(
        size_m <= i32::MAX as usize && size_n <= i32::MAX as usize && size_k <= i32::MAX as usize,
        "Kimi Marlin WNA16 MNK exceeds i32"
    );
    ensure!(
        routing.max_padded_tokens <= i32::MAX as usize
            && workspace.locks.len() <= i32::MAX as usize,
        "Kimi Marlin WNA16 metadata exceeds i32"
    );
    ensure!(
        weight_packed_uint4b8.len() > 0 && weight_scale_permuted.len() > 0,
        "Kimi Marlin WNA16 weight package must be non-empty"
    );
    let lock_len = workspace.locks.len();
    let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.data.device_ptr_mut(&ctx.stream);
    let (c_tmp_ptr, _c_tmp_guard) = workspace.c_tmp.device_ptr_mut(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight_packed_uint4b8.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = weight_scale_permuted.device_ptr(&ctx.stream);
    let (locks_ptr, _locks_guard) = workspace.locks.device_ptr_mut(&ctx.stream);
    let (sorted_ptr, _sorted_guard) = routing.sorted_token_ids.device_ptr(&ctx.stream);
    let (expert_ids_ptr, _expert_ids_guard) = routing.expert_ids.device_ptr(&ctx.stream);
    let (num_tokens_ptr, _num_tokens_guard) =
        routing.num_tokens_post_padded.device_ptr(&ctx.stream);
    let (topk_ptr, _topk_guard) = topk_weight.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_wna16_gemm_cuda(
            input_ptr as *const ffi::Half,
            output_ptr as *mut ffi::Half,
            c_tmp_ptr as *mut f32,
            weight_ptr as *const u8,
            scale_ptr as *const ffi::Half,
            locks_ptr as *mut i32,
            sorted_ptr as *const i32,
            expert_ids_ptr as *const i32,
            num_tokens_ptr as *const i32,
            topk_ptr as *const f32,
            lock_len as i32,
            routing.max_padded_tokens as i32,
            routing.block_size as i32,
            top_k as i32,
            mul_topk_weights,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            0,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_int4_grouped_w1_w3(
    ctx: &DeviceContext,
    plan: &mut KimiGroupedW1W3Plan<'_>,
    weights: &KimiInt4ExpertWeights<'_>,
) -> Result<()> {
    plan.validate(weights)?;
    let (x_ptr, _x_guard) = plan.expert_hidden.data.device_ptr(&ctx.stream);
    let (w1_ptr, _w1_guard) = weights.w1_gate.weight_packed.device_ptr(&ctx.stream);
    let (w1_scale_ptr, _w1_scale_guard) = weights.w1_gate.weight_scale.device_ptr(&ctx.stream);
    let (w1_shape_ptr, _w1_shape_guard) = weights.w1_gate.weight_shape.device_ptr(&ctx.stream);
    let (w3_ptr, _w3_guard) = weights.w3_up.weight_packed.device_ptr(&ctx.stream);
    let (w3_scale_ptr, _w3_scale_guard) = weights.w3_up.weight_scale.device_ptr(&ctx.stream);
    let (w3_shape_ptr, _w3_shape_guard) = weights.w3_up.weight_shape.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = plan.route.expert_indptr.device_ptr(&ctx.stream);
    let (gate_ptr, _gate_guard) = plan.gate_out.data.device_ptr_mut(&ctx.stream);
    let (up_ptr, _up_guard) = plan.up_out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::kimi_int4_grouped_w1_w3_cuda(
            x_ptr as *const ffi::Half,
            w1_ptr as *const u8,
            w1_scale_ptr as *const ffi::Half,
            w1_shape_ptr as *const i32,
            w3_ptr as *const u8,
            w3_scale_ptr as *const ffi::Half,
            w3_shape_ptr as *const i32,
            expert_ptr as *const u32,
            gate_ptr as *mut ffi::Half,
            up_ptr as *mut ffi::Half,
            plan.route.routed_tokens as i32,
            KIMI_K2_HIDDEN as i32,
            KIMI_K2_EXPERT_INTERMEDIATE as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_int4_grouped_w2_swiglu(
    ctx: &DeviceContext,
    plan: &mut KimiGroupedW2SwiGluPlan<'_>,
    weights: &KimiInt4ExpertWeights<'_>,
) -> Result<()> {
    plan.validate(weights)?;
    let (gate_ptr, _gate_guard) = plan.gate.data.device_ptr(&ctx.stream);
    let (up_ptr, _up_guard) = plan.up.data.device_ptr(&ctx.stream);
    let (w2_ptr, _w2_guard) = weights.w2_down.weight_packed.device_ptr(&ctx.stream);
    let (w2_scale_ptr, _w2_scale_guard) = weights.w2_down.weight_scale.device_ptr(&ctx.stream);
    let (w2_shape_ptr, _w2_shape_guard) = weights.w2_down.weight_shape.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = plan.route.expert_indptr.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = plan.expert_output.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::kimi_int4_grouped_w2_swiglu_cuda(
            gate_ptr as *const ffi::Half,
            up_ptr as *const ffi::Half,
            w2_ptr as *const u8,
            w2_scale_ptr as *const ffi::Half,
            w2_shape_ptr as *const i32,
            expert_ptr as *const u32,
            out_ptr as *mut ffi::Half,
            plan.route.routed_tokens as i32,
            KIMI_K2_EXPERT_INTERMEDIATE as i32,
            KIMI_K2_HIDDEN as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_moe_build_expert_major_route<'a>(
    ctx: &DeviceContext,
    batch_size: usize,
    active_tokens: usize,
    global_expert_start: usize,
    topk_idx: &CudaSlice<i32>,
    workspace: &'a mut KimiExpertMajorRouteWorkspace,
) -> Result<KimiExpertMajorRouting<'a>> {
    ensure!(
        batch_size > 0,
        "Kimi expert-major route batch_size must be positive"
    );
    ensure!(
        active_tokens >= batch_size,
        "active_tokens {} must cover batch_size {}",
        active_tokens,
        batch_size
    );
    ensure!(
        global_expert_start + KIMI_K2_LOCAL_EXPERTS <= KIMI_K2_ROUTED_EXPERTS,
        "global expert range [{}..{}) exceeds {} experts",
        global_expert_start,
        global_expert_start + KIMI_K2_LOCAL_EXPERTS,
        KIMI_K2_ROUTED_EXPERTS
    );
    workspace.validate_for(active_tokens)?;
    let route_capacity = KimiExpertMajorRouteWorkspace::route_capacity(active_tokens);
    ensure!(
        topk_idx.len() >= route_capacity,
        "topk_idx too small: have {}, need {}",
        topk_idx.len(),
        route_capacity
    );

    {
        let (topk_idx_ptr, _topk_idx_guard) = topk_idx.device_ptr(&ctx.stream);
        let (pos_to_token_ptr, _pos_to_token_guard) =
            workspace.pos_to_token.device_ptr_mut(&ctx.stream);
        let (token_topk_to_pos_ptr, _token_topk_to_pos_guard) =
            workspace.token_topk_to_pos.device_ptr_mut(&ctx.stream);
        let (expert_indptr_ptr, _expert_indptr_guard) =
            workspace.expert_indptr.device_ptr_mut(&ctx.stream);
        let (expert_cursor_ptr, _expert_cursor_guard) =
            workspace.expert_cursor.device_ptr_mut(&ctx.stream);
        let (local_count_ptr, _local_count_guard) =
            workspace.local_count.device_ptr_mut(&ctx.stream);

        let result = unsafe {
            ffi::kimi_moe_expert_major_route_cuda(
                topk_idx_ptr as *const i32,
                pos_to_token_ptr as *mut i32,
                token_topk_to_pos_ptr as *mut i32,
                expert_indptr_ptr as *mut u32,
                expert_cursor_ptr as *mut u32,
                local_count_ptr as *mut u32,
                active_tokens as i32,
                KIMI_K2_TOPK as i32,
                global_expert_start as i32,
                KIMI_K2_LOCAL_EXPERTS as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    let route = KimiExpertMajorRoute {
        batch_size,
        active_tokens,
        routed_tokens: route_capacity,
        expert_indptr: &workspace.expert_indptr,
    };
    route.validate()?;
    Ok(KimiExpertMajorRouting {
        route,
        pos_to_token: &workspace.pos_to_token,
        token_topk_to_pos: &workspace.token_topk_to_pos,
        local_count: &workspace.local_count,
        global_expert_start,
    })
}

pub fn kimi_moe_expand_to_expert_major(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    routing: &KimiExpertMajorRouting<'_>,
    expert_major_hidden: &mut HiddenStates,
) -> Result<()> {
    routing.route.validate()?;
    ensure!(
        hidden.seq_len >= routing.route.active_tokens,
        "hidden seq_len {} must cover active_tokens {}",
        hidden.seq_len,
        routing.route.active_tokens
    );
    validate_hidden_states(
        "expert_major_expand.input",
        hidden,
        hidden.hidden_dim,
        hidden.seq_len,
    )?;
    validate_hidden_states(
        "expert_major_expand.output",
        expert_major_hidden,
        hidden.hidden_dim,
        routing.route.routed_tokens,
    )?;
    let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
    let (pos_ptr, _pos_guard) = routing.pos_to_token.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = expert_major_hidden.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_moe_expand_to_expert_major_cuda(
            hidden_ptr as *const ffi::Half,
            pos_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            hidden.hidden_dim as i32,
            routing.route.routed_tokens as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_moe_reduce_expert_major_f32(
    ctx: &DeviceContext,
    expert_major_output: &HiddenStates,
    topk_weight: &CudaSlice<f32>,
    routing: &KimiExpertMajorRouting<'_>,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    routing.route.validate()?;
    validate_hidden_states(
        "expert_major_reduce.input",
        expert_major_output,
        expert_major_output.hidden_dim,
        routing.route.routed_tokens,
    )?;
    let route_entries = routing.route.active_tokens * KIMI_K2_TOPK;
    ensure!(
        topk_weight.len() >= route_entries,
        "topk_weight too small: have {}, need {}",
        topk_weight.len(),
        route_entries
    );
    let output_elems = routing.route.active_tokens * expert_major_output.hidden_dim;
    ensure!(
        out.len() >= output_elems,
        "Kimi expert-major reduce output too small: have {}, need {}",
        out.len(),
        output_elems
    );

    let (expert_output_ptr, _expert_output_guard) =
        expert_major_output.data.device_ptr(&ctx.stream);
    let (topk_weight_ptr, _topk_weight_guard) = topk_weight.device_ptr(&ctx.stream);
    let (map_ptr, _map_guard) = routing.token_topk_to_pos.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_moe_reduce_expert_major_f32_cuda(
            expert_output_ptr as *const ffi::Half,
            topk_weight_ptr as *const f32,
            map_ptr as *const i32,
            out_ptr as *mut f32,
            routing.route.active_tokens as i32,
            expert_major_output.hidden_dim as i32,
            KIMI_K2_TOPK as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[must_use]
pub const fn packed_int4_cols(cols: usize) -> usize {
    cols.div_ceil(2)
}

fn validate_marlin_block_size(block_size: usize) -> Result<()> {
    ensure!(
        block_size == 8 || (block_size >= 16 && block_size <= 64 && block_size.is_multiple_of(16)),
        "Kimi Marlin block_size must be 8 or a multiple of 16 in [16, 64], got {}",
        block_size
    );
    Ok(())
}

fn validate_global_expert_start(global_expert_start: usize) -> Result<()> {
    ensure!(
        global_expert_start + KIMI_K2_LOCAL_EXPERTS <= KIMI_K2_ROUTED_EXPERTS,
        "global expert range [{}..{}) exceeds {} routed experts",
        global_expert_start,
        global_expert_start + KIMI_K2_LOCAL_EXPERTS,
        KIMI_K2_ROUTED_EXPERTS
    );
    Ok(())
}

fn marlin_padded_route_capacity(active_tokens: usize, block_size: usize) -> Result<usize> {
    validate_marlin_block_size(block_size)?;
    let route_elems = active_tokens
        .checked_mul(KIMI_K2_TOPK)
        .ok_or_else(|| anyhow::anyhow!("active_tokens * topk overflow"))?;
    let max_padding = KIMI_K2_LOCAL_EXPERTS
        .checked_mul(block_size - 1)
        .ok_or_else(|| anyhow::anyhow!("local_experts * (block_size - 1) overflow"))?;
    route_elems
        .checked_add(max_padding)
        .ok_or_else(|| anyhow::anyhow!("Marlin padded route capacity overflow"))
}

fn ensure_role(weight: &KimiInt4Weight<'_>, expected: KimiInt4ExpertRole) -> Result<()> {
    ensure!(
        weight.manifest.role == expected,
        "expert role must be {:?}, got {:?}",
        expected,
        weight.manifest.role
    );
    Ok(())
}

fn validate_hidden_states(
    name: &str,
    states: &HiddenStates,
    hidden_dim: usize,
    seq_len: usize,
) -> Result<()> {
    ensure!(
        states.hidden_dim == hidden_dim,
        "{name} hidden_dim must be {hidden_dim}, got {}",
        states.hidden_dim
    );
    ensure!(
        states.seq_len == seq_len,
        "{name} seq_len must be {seq_len}, got {}",
        states.seq_len
    );
    ensure!(
        states.data.len() >= hidden_dim * seq_len,
        "{name} storage len must cover {}, got {}",
        hidden_dim * seq_len,
        states.data.len()
    );
    Ok(())
}

fn validate_cutlass_projection(
    name: &str,
    workspace: &KimiCutlassInt4GroupedWorkspace,
    route: &KimiExpertMajorRoute<'_>,
    input: &HiddenStates,
    weight: &KimiInt4Weight<'_>,
    output: &HiddenStates,
) -> Result<()> {
    weight.validate()?;
    route.validate()?;
    workspace.validate_for(
        route.routed_tokens,
        weight.manifest.logical_shape.in_dim,
        weight.manifest.logical_shape.out_dim,
    )?;
    validate_hidden_states(
        &format!("{name}.input"),
        input,
        weight.manifest.logical_shape.in_dim,
        route.routed_tokens,
    )?;
    validate_hidden_states(
        &format!("{name}.output"),
        output,
        weight.manifest.logical_shape.out_dim,
        route.routed_tokens,
    )?;
    ensure!(
        weight.manifest.nibble_order == KimiInt4NibbleOrder::LowThenHigh,
        "{} requires low-then-high packed INT4 before CUTLASS reorder, got {}",
        weight.manifest.role.label(),
        weight.manifest.nibble_order.label()
    );
    Ok(())
}

fn hidden_spec(hidden_dim: usize, tokens: usize) -> TensorSpec {
    TensorSpec::named(
        "bf16",
        "expert_major",
        [
            AxisSpec::named("routed_token", tokens),
            AxisSpec::named("hidden", hidden_dim),
        ],
    )
}

pub struct KimiSwiGluPlan<'a> {
    pub route: KimiExpertMajorRoute<'a>,
    pub gate: &'a HiddenStates,
    pub up: &'a HiddenStates,
    pub activated: &'a mut HiddenStates,
}

impl KimiSwiGluPlan<'_> {
    pub fn validate(&self) -> Result<()> {
        self.route.validate()?;
        validate_hidden_states(
            "swiglu.gate",
            self.gate,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )?;
        validate_hidden_states(
            "swiglu.up",
            self.up,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )?;
        validate_hidden_states(
            "swiglu.activated",
            self.activated,
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.route.routed_tokens,
        )
    }

    #[must_use]
    pub fn manifest_call(&self) -> KernelCall {
        KernelCall::new(
            "kimi_k2.moe.swiglu_silu_mul",
            "Kimi-K2 expert SwiGLU activation between W1/W3 and W2",
        )
        .input(
            "gate",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .input(
            "up",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .output(
            "activated",
            hidden_spec(KIMI_K2_EXPERT_INTERMEDIATE, self.route.routed_tokens),
        )
        .attr("local_experts", KIMI_K2_LOCAL_EXPERTS.to_string())
        .attr("topk", KIMI_K2_TOPK.to_string())
        .attr("batch_size", self.route.batch_size.to_string())
        .attr("active_tokens", self.route.active_tokens.to_string())
        .attr("routed_tokens", self.route.routed_tokens.to_string())
        .attr("layout", "expert_major_routed_tokens".to_string())
        .attr("activation", "silu_gate_mul_up".to_string())
        .attr("dtype", "bf16".to_string())
        .attr("accumulator_dtype", "f32".to_string())
        .attr("cuda_graph_ready", "yes".to_string())
        .attr("kernel", "elementwise.silu_mul_triton_aot_cuda".to_string())
    }
}

/// SwiGLU activation between INT4 grouped W1/W3 and W2 for Kimi-K2 routed experts.
///
/// Computes `activated[i] = silu(gate[i]) * up[i]` element-wise over expert-major
/// `[routed_tokens, KIMI_K2_EXPERT_INTERMEDIATE]` BF16 buffers, reusing the shared
/// `silu_mul_triton_aot_cuda` kernel. Output is layer-resident scratch so the
/// follow-on W2 grouped INT4 GEMM consumes BF16 activations without copies.
pub fn kimi_swiglu_silu_mul(ctx: &DeviceContext, plan: &mut KimiSwiGluPlan<'_>) -> Result<()> {
    plan.validate()?;
    let n = KIMI_K2_EXPERT_INTERMEDIATE * plan.route.routed_tokens;
    if n == 0 {
        return Ok(());
    }
    let (gate_ptr, _gate_guard) = plan.gate.data.device_ptr(&ctx.stream);
    let (up_ptr, _up_guard) = plan.up.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = plan.activated.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_triton_aot_cuda(
            gate_ptr as *const ffi::Half,
            up_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn validate_ep_rank(ep_rank: usize) -> Result<()> {
    if ep_rank < KIMI_K2_EP_WORLD {
        Ok(())
    } else {
        bail!(
            "Kimi-K2 EP rank must be < {}, got {}",
            KIMI_K2_EP_WORLD,
            ep_rank
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep8_w1_manifest_shapes_cover_compressed_tensors_metadata() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W1Gate,
            3,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        manifest.validate().expect("manifest should be valid");
        assert_eq!(manifest.local_experts, 48);
        assert_eq!(manifest.local_expert_offset, 144);
        assert_eq!(manifest.packed_shape.elements(), 48 * 2048 * (7168 / 2));
        assert_eq!(manifest.scale_shape.elements(), 48 * 2048 * (7168 / 32));
        assert_eq!(manifest.weight_shape_entries, 96);
    }

    #[test]
    fn ep8_w2_manifest_uses_down_projection_shape() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W2Down,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        manifest.validate().expect("manifest should be valid");
        assert_eq!(
            manifest.logical_shape,
            KimiInt4LogicalShape {
                out_dim: KIMI_K2_HIDDEN,
                in_dim: KIMI_K2_EXPERT_INTERMEDIATE,
            }
        );
        assert_eq!(manifest.packed_shape.elements(), 48 * 7168 * (2048 / 2));
        assert_eq!(manifest.scale_shape.elements(), 48 * 7168 * (2048 / 32));
    }

    #[test]
    fn int4_scale_specs_distinguish_checkpoint_cutlass_and_marlin_layouts() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W1Gate,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        let checkpoint = manifest.weight_scale_checkpoint_spec();
        assert_eq!(checkpoint.layout, "expert_major_group_scale_checkpoint");
        assert_eq!(checkpoint.axes[1].name, "out");
        assert_eq!(checkpoint.axes[2].name, "in_group");

        let cutlass = manifest.weight_scale_cutlass_example69_spec();
        assert_eq!(cutlass.layout, "expert_major_group_scale_cutlass_example69");
        assert_eq!(cutlass.axes[1].name, "in_group");
        assert_eq!(cutlass.axes[2].name, "out");

        let marlin = manifest.weight_scale_marlin_permuted_spec();
        assert_eq!(
            marlin.layout,
            "expert_major_group_scale_marlin_group_major_perm64"
        );
        assert_eq!(marlin.axes[1].name, "in_group");
        assert_eq!(marlin.axes[2].name, "out");
    }

    #[test]
    fn int4_packed_specs_distinguish_checkpoint_cutlass_and_marlin_layouts() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W2Down,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        let checkpoint = manifest.weight_packed_checkpoint_spec();
        assert_eq!(
            checkpoint.layout,
            "expert_major_int4_packed_checkpoint_offset_binary"
        );
        assert_eq!(checkpoint.axes[1].name, "out");
        assert_eq!(checkpoint.axes[2].name, "packed_in_over_2");

        let cutlass = manifest.weight_packed_cutlass_example69_spec();
        assert_eq!(
            cutlass.layout,
            "expert_major_int4_packed_cutlass_example69_signed_reordered"
        );
        assert_eq!(cutlass.axes[1].name, "cutlass_reordered_out");
        assert_eq!(cutlass.axes[2].name, "cutlass_reordered_packed_in_over_2");

        let marlin = manifest.weight_packed_marlin_uint4b8_spec();
        assert_eq!(
            marlin.layout,
            "expert_major_int4_packed_marlin_uint4b8_noact"
        );
        assert_eq!(marlin.dtype, "u32");
        assert_eq!(marlin.axes[1].name, "in_tile16");
        assert_eq!(marlin.axes[1].size, KIMI_K2_EXPERT_INTERMEDIATE / 16);
        assert_eq!(marlin.axes[2].name, "out_x2");
        assert_eq!(marlin.axes[2].size, KIMI_K2_HIDDEN * 2);
        assert_eq!(
            manifest.marlin_packed_u32_elements() * std::mem::size_of::<u32>(),
            manifest.packed_shape.elements()
        );
    }

    #[test]
    fn int4_offset_binary_nibbles_decode_to_signed_by_subtracting_eight() {
        let decode = |byte: u8, col: usize| -> i8 {
            let unsigned = if col.is_multiple_of(2) {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            i8::try_from(unsigned).expect("nibble") - 8
        };

        for signed_even in -8i8..=7 {
            for signed_odd in -8i8..=7 {
                let even = u8::try_from(signed_even + 8).expect("even nibble");
                let odd = u8::try_from(signed_odd + 8).expect("odd nibble");
                let byte = even | (odd << 4);
                assert_eq!(decode(byte, 0), signed_even);
                assert_eq!(decode(byte, 1), signed_odd);
                assert_eq!(i16::from(even) - i16::from(signed_even), 8);
                assert_eq!(i16::from(odd) - i16::from(signed_odd), 8);
            }
        }
    }

    #[test]
    fn route_layout_accepts_multi_batch_active_tokens() {
        assert_eq!(KimiExpertMajorRoute::max_routed_tokens(5), 40);
    }

    #[test]
    fn marlin_route_capacity_matches_vllm_ignore_invalid_bound() {
        let active_tokens = 7;
        let block_size = 8;
        let route_elems = active_tokens * KIMI_K2_TOPK;
        let capacity = marlin_padded_route_capacity(active_tokens, block_size).expect("capacity");
        assert_eq!(
            capacity,
            route_elems + KIMI_K2_LOCAL_EXPERTS * (block_size - 1)
        );
        assert_eq!(capacity.div_ceil(block_size), 49);
    }

    #[test]
    fn swiglu_manifest_call_carries_expert_major_layout() {
        let indptr = std::iter::repeat_n(0u32, KIMI_K2_LOCAL_EXPERTS + 1).collect::<Vec<_>>();
        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let indptr_dev = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
        let gate = crate::tensor::HiddenStates::zeros(&ctx, KIMI_K2_EXPERT_INTERMEDIATE, 0)
            .expect("gate buffer");
        let up = crate::tensor::HiddenStates::zeros(&ctx, KIMI_K2_EXPERT_INTERMEDIATE, 0)
            .expect("up buffer");
        let mut activated =
            crate::tensor::HiddenStates::zeros(&ctx, KIMI_K2_EXPERT_INTERMEDIATE, 0)
                .expect("out buffer");
        let plan = KimiSwiGluPlan {
            route: KimiExpertMajorRoute {
                batch_size: 1,
                active_tokens: 1,
                routed_tokens: 0,
                expert_indptr: &indptr_dev,
            },
            gate: &gate,
            up: &up,
            activated: &mut activated,
        };
        let call = plan.manifest_call();
        let attrs: std::collections::HashMap<&str, &str> = call
            .attrs
            .iter()
            .map(|a| (a.name.as_str(), a.value.as_str()))
            .collect();
        assert_eq!(attrs.get("activation"), Some(&"silu_gate_mul_up"));
        assert_eq!(attrs.get("layout"), Some(&"expert_major_routed_tokens"));
        assert_eq!(attrs.get("dtype"), Some(&"bf16"));
    }

    #[test]
    #[ignore = "H20-only: verifies vLLM Marlin WNA16 route alignment metadata on device"]
    fn h20_kimi_marlin_align_block_size_matches_vllm_contract() {
        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let batch_size = 4usize;
        let active_tokens = 7usize;
        let block_size = 8usize;
        let global_start = 96usize;
        let topk = KIMI_K2_TOPK;
        let route_elems = active_tokens * topk;
        let topk_host = vec![
            96, 97, 12, 143, 144, 98, 380, 99, 97, 96, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 96, 96, 96, 96, 96, 96, 96, 96, 120, 121, 122, 123, 124, 125,
            126, 127, 143, 143, 143, 143, 143, 143, 143, 143, 0, 383, 95, 144, 145, 146, 147, 148,
        ];
        assert_eq!(topk_host.len(), route_elems);

        let topk_dev = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
        let mut workspace =
            KimiMarlinRouteWorkspace::new(&ctx, active_tokens, block_size).expect("workspace");
        let routing = kimi_moe_marlin_align_block_size(
            &ctx,
            &mut workspace,
            &topk_dev,
            batch_size,
            active_tokens,
            global_start,
        )
        .expect("align");

        let num_tokens = ctx
            .stream
            .clone_dtoh(routing.num_tokens_post_padded)
            .expect("num_tokens D2H");
        let total = usize::try_from(num_tokens[0]).expect("nonnegative padded tokens");
        assert!(total.is_multiple_of(block_size));

        let sorted = ctx
            .stream
            .clone_dtoh(routing.sorted_token_ids)
            .expect("sorted D2H");
        let expert_ids = ctx
            .stream
            .clone_dtoh(routing.expert_ids)
            .expect("expert_ids D2H");

        let mut expected_sorted = Vec::<i32>::new();
        let mut expected_expert_ids = Vec::<i32>::new();
        let sentinel = i32::try_from(route_elems).expect("route sentinel");
        for local_expert in 0..KIMI_K2_LOCAL_EXPERTS {
            let global_expert = global_start + local_expert;
            let mut routes = topk_host
                .iter()
                .enumerate()
                .filter_map(|(route_offset, &expert)| {
                    (usize::try_from(expert).ok() == Some(global_expert))
                        .then(|| i32::try_from(route_offset).expect("route offset"))
                })
                .collect::<Vec<_>>();
            if routes.is_empty() {
                continue;
            }
            let padded = routes.len().div_ceil(block_size) * block_size;
            expected_expert_ids.extend(std::iter::repeat_n(
                i32::try_from(local_expert).expect("local expert"),
                padded / block_size,
            ));
            routes.extend(std::iter::repeat_n(sentinel, padded - routes.len()));
            expected_sorted.extend(routes);
        }

        assert_eq!(total, expected_sorted.len());
        assert_eq!(&sorted[..total], expected_sorted.as_slice());
        assert_eq!(
            &expert_ids[..expected_expert_ids.len()],
            expected_expert_ids.as_slice()
        );

        let call = routing.manifest_call();
        let attrs: std::collections::HashMap<&str, &str> = call
            .attrs
            .iter()
            .map(|a| (a.name.as_str(), a.value.as_str()))
            .collect();
        assert_eq!(attrs.get("device_resident_metadata"), Some(&"true"));
        assert_eq!(attrs.get("decode_step_d2h"), Some(&"forbidden"));
        assert_eq!(attrs.get("sentinel_token_id"), Some(&"56"));
    }

    #[test]
    fn swiglu_gpu_kernel_matches_silu_mul_reference() {
        use half::bf16;
        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");

        let routed_tokens = 4usize;
        let intermediate = KIMI_K2_EXPERT_INTERMEDIATE;
        let n = routed_tokens * intermediate;

        let mut gate_host = Vec::with_capacity(n);
        let mut up_host = Vec::with_capacity(n);
        for i in 0..n {
            let g = ((i as f32) * 0.013).sin();
            let u = ((i as f32) * 0.017).cos();
            gate_host.push(bf16::from_f32(g));
            up_host.push(bf16::from_f32(u));
        }

        let mut gate = crate::tensor::HiddenStates::zeros(&ctx, intermediate, routed_tokens)
            .expect("gate alloc");
        let mut up = crate::tensor::HiddenStates::zeros(&ctx, intermediate, routed_tokens)
            .expect("up alloc");
        let mut activated = crate::tensor::HiddenStates::zeros(&ctx, intermediate, routed_tokens)
            .expect("out alloc");
        ctx.stream
            .memcpy_htod(&gate_host, &mut gate.data)
            .expect("gate H2D");
        ctx.stream
            .memcpy_htod(&up_host, &mut up.data)
            .expect("up H2D");

        let indptr = std::iter::repeat_n(0u32, KIMI_K2_LOCAL_EXPERTS + 1).collect::<Vec<_>>();
        let indptr_dev = ctx.stream.clone_htod(&indptr).expect("indptr H2D");

        {
            let mut plan = KimiSwiGluPlan {
                route: KimiExpertMajorRoute {
                    batch_size: 1,
                    active_tokens: routed_tokens,
                    routed_tokens,
                    expert_indptr: &indptr_dev,
                },
                gate: &gate,
                up: &up,
                activated: &mut activated,
            };
            kimi_swiglu_silu_mul(&ctx, &mut plan).expect("swiglu launch");
        }
        ctx.sync().expect("sync");

        let out_host: Vec<bf16> = ctx.stream.clone_dtoh(&activated.data).expect("D2H");
        ctx.sync().expect("sync");

        for i in 0..n {
            let g = gate_host[i].to_f32();
            let u = up_host[i].to_f32();
            let silu_g = g / (1.0 + (-g).exp());
            let expected = bf16::from_f32(bf16::from_f32(silu_g).to_f32() * u).to_f32();
            let actual = out_host[i].to_f32();
            assert!(
                (actual - expected).abs() <= 1e-3,
                "i={i} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    #[ignore = "H20-only: verifies vLLM Marlin scale layout packer on device"]
    fn h20_kimi_marlin_scale_reorder_matches_vllm_permute() {
        use half::bf16;

        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let local_experts = KIMI_K2_LOCAL_EXPERTS;
        let group_size = KIMI_K2_INT4_GROUP_SIZE;
        let in_dim = 64usize;
        let out_dim = 64usize;
        let scale_k = in_dim / group_size;
        let elements_per_expert = out_dim * scale_k;
        assert_eq!(elements_per_expert % 64, 0);

        let scale_value = |expert: usize, row: usize, group: usize| -> bf16 {
            bf16::from_f32(expert as f32 * 0.25 + row as f32 * 0.01 + group as f32 * 0.125)
        };
        let mut checkpoint = vec![bf16::ZERO; local_experts * elements_per_expert];
        for expert in 0..local_experts {
            for row in 0..out_dim {
                for group in 0..scale_k {
                    checkpoint[expert * elements_per_expert + row * scale_k + group] =
                        scale_value(expert, row, group);
                }
            }
        }

        let checkpoint_dev = ctx.stream.clone_htod(&checkpoint).expect("scale H2D");
        let mut marlin_dev = ctx
            .stream
            .alloc_zeros::<bf16>(checkpoint.len())
            .expect("marlin scale alloc");
        {
            let (src_ptr, _src_guard) = checkpoint_dev.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = marlin_dev.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_marlin_int4_reorder_scale_cuda(
                    src_ptr as *const crate::ffi::Half,
                    dst_ptr as *mut crate::ffi::Half,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("marlin scale reorder");
        }
        let got = ctx.stream.clone_dtoh(&marlin_dev).expect("scale D2H");
        ctx.sync().expect("sync");

        let marlin_scale_perm = |offset: usize| -> usize { offset / 8 + 8 * (offset % 8) };
        for expert in [0usize, 7, local_experts - 1] {
            for flat in 0..elements_per_expert {
                let source_flat = (flat / 64) * 64 + marlin_scale_perm(flat % 64);
                let group = source_flat / out_dim;
                let row = source_flat - group * out_dim;
                let idx = expert * elements_per_expert + flat;
                let expected = checkpoint[expert * elements_per_expert + row * scale_k + group];
                assert_eq!(
                    got[idx].to_bits(),
                    expected.to_bits(),
                    "expert={expert} flat={flat} row={row} group={group}"
                );
            }
        }
    }

    #[test]
    #[ignore = "H20-only: verifies vLLM no-actorder Marlin weight repack layout on device"]
    fn h20_kimi_marlin_weight_repack_matches_vllm_noact_layout() {
        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let local_experts = KIMI_K2_LOCAL_EXPERTS;
        let group_size = KIMI_K2_INT4_GROUP_SIZE;
        let in_dim = 64usize;
        let out_dim = 64usize;
        let pack_factor = 8usize;
        let tile_k = 16usize;
        let tile_n = 64usize;
        let k_packed_cols = in_dim / pack_factor;
        let k_tiles = in_dim / tile_k;
        let n_tiles = out_dim / tile_n;
        let words_per_expert = out_dim * k_packed_cols;
        let marlin_words_per_expert = k_tiles * out_dim * 2;
        assert_eq!(words_per_expert, marlin_words_per_expert);

        let nibble = |expert: usize, row: usize, col: usize| -> u32 {
            ((expert * 3 + row * 5 + col * 7) & 0x0f) as u32
        };
        let mut checkpoint = vec![0u32; local_experts * words_per_expert];
        for expert in 0..local_experts {
            for row in 0..out_dim {
                for k_word in 0..k_packed_cols {
                    let mut word = 0u32;
                    for pos in 0..pack_factor {
                        word |= nibble(expert, row, k_word * pack_factor + pos) << (pos * 4);
                    }
                    checkpoint[expert * words_per_expert + row * k_packed_cols + k_word] = word;
                }
            }
        }

        let mut expected = vec![0u32; checkpoint.len()];
        let tc_offsets = [0usize, 1, 8, 9];
        let pack_idx = [0usize, 2, 4, 6, 1, 3, 5, 7];
        for expert in 0..local_experts {
            let checkpoint_base = expert * words_per_expert;
            let marlin_base = expert * marlin_words_per_expert;
            for k_tile in 0..k_tiles {
                for n_tile in 0..n_tiles {
                    let mut sh_stage = vec![0u32; tile_n * (tile_k / pack_factor)];
                    for k_id in 0..(tile_k / pack_factor) {
                        for n in 0..tile_n {
                            sh_stage[k_id * tile_n + n] = checkpoint[checkpoint_base
                                + (n_tile * tile_n + n) * k_packed_cols
                                + k_tile * (tile_k / pack_factor)
                                + k_id];
                        }
                    }
                    for warp_id in 0..4usize {
                        for th_id in 0..32usize {
                            let tc_col = th_id / 4;
                            let tc_row = (th_id % 4) * 2;
                            let cur_n = warp_id * 16 + tc_col;
                            let b1_vals = [sh_stage[cur_n], sh_stage[cur_n + tile_n]];
                            let b2_vals = [sh_stage[cur_n + 8], sh_stage[cur_n + 8 + tile_n]];

                            let mut vals = [0u32; 8];
                            for i in 0..4usize {
                                let cur_elem = tc_row + tc_offsets[i];
                                let cur_int = cur_elem / pack_factor;
                                let cur_pos = cur_elem % pack_factor;
                                vals[i] = (b1_vals[cur_int] >> (cur_pos * 4)) & 0x0f;
                                vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * 4)) & 0x0f;
                            }

                            let mut packed = 0u32;
                            for i in 0..8usize {
                                packed |= vals[pack_idx[i]] << (i * 4);
                            }
                            let tile_size = tile_k * tile_n / pack_factor;
                            let out_offset = (k_tile * n_tiles + n_tile) * tile_size;
                            expected[marlin_base + out_offset + th_id * 4 + warp_id] = packed;
                        }
                    }
                }
            }
        }

        let checkpoint_dev = ctx.stream.clone_htod(&checkpoint).expect("weight H2D");
        let mut marlin_dev = ctx
            .stream
            .alloc_zeros::<u32>(checkpoint.len())
            .expect("marlin weight alloc");
        {
            let (src_ptr, _src_guard) = checkpoint_dev.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = marlin_dev.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_marlin_int4_reorder_weight_cuda(
                    src_ptr as *const u8,
                    dst_ptr as *mut u8,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("marlin weight reorder");
        }
        let got = ctx.stream.clone_dtoh(&marlin_dev).expect("weight D2H");
        assert_eq!(got, expected);
    }

    #[test]
    #[ignore = "H20-only: documents why CUTLASS example69 is not a Kimi group-size-32 correctness path"]
    fn h20_kimi_cutlass_int4_example69_rejects_per32_scale_semantics() {
        use half::bf16;

        #[derive(Clone, Copy)]
        struct ProbeCase {
            row: usize,
            col: usize,
            unsigned_nibble: u8,
        }

        let cases = [
            ProbeCase {
                row: 0,
                col: 0,
                unsigned_nibble: 9,
            },
            ProbeCase {
                row: 1,
                col: 1,
                unsigned_nibble: 7,
            },
            ProbeCase {
                row: 2,
                col: 31,
                unsigned_nibble: 15,
            },
            ProbeCase {
                row: 3,
                col: 32,
                unsigned_nibble: 0,
            },
            ProbeCase {
                row: 4,
                col: 33,
                unsigned_nibble: 14,
            },
            ProbeCase {
                row: 5,
                col: 64,
                unsigned_nibble: 10,
            },
        ];

        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let local_experts = KIMI_K2_LOCAL_EXPERTS;
        let group_size = KIMI_K2_INT4_GROUP_SIZE;
        let in_dim = KIMI_K2_HIDDEN;
        let out_dim = KIMI_K2_EXPERT_INTERMEDIATE;
        let routed_tokens = cases.len();
        let indptr = (0..=local_experts)
            .map(|expert| if expert == 0 { 0 } else { routed_tokens as u32 })
            .collect::<Vec<_>>();

        let mut input_host = vec![bf16::ZERO; routed_tokens * in_dim];
        for (token, case) in cases.iter().enumerate() {
            input_host[token * in_dim + case.col] = bf16::from_f32(1.0);
        }

        let mut packed_host = vec![0x88u8; local_experts * out_dim * (in_dim / 2)];
        let scale_value = |row: usize, group: usize| -> f32 {
            0.25 + row as f32 * 0.125 + group as f32 * 0.03125
        };
        let mut scale_host =
            vec![bf16::from_f32(1.0); local_experts * out_dim * (in_dim / group_size)];
        for row in 0..out_dim {
            for group in 0..(in_dim / group_size) {
                scale_host[row * (in_dim / group_size) + group] =
                    bf16::from_f32(scale_value(row, group));
            }
        }
        for case in cases {
            let byte_idx = case.row * (in_dim / 2) + case.col / 2;
            let byte = &mut packed_host[byte_idx];
            if case.col % 2 == 0 {
                *byte = (*byte & 0xf0) | (case.unsigned_nibble & 0x0f);
            } else {
                *byte = (*byte & 0x0f) | ((case.unsigned_nibble & 0x0f) << 4);
            }
        }

        let input = ctx.stream.clone_htod(&input_host).expect("input H2D");
        let packed_offset_binary = ctx.stream.clone_htod(&packed_host).expect("weight H2D");
        let mut packed_reordered = ctx
            .stream
            .alloc_zeros::<u8>(packed_host.len())
            .expect("reordered alloc");
        let scale_checkpoint = ctx.stream.clone_htod(&scale_host).expect("scale H2D");
        let mut scale = ctx
            .stream
            .alloc_zeros::<bf16>(scale_host.len())
            .expect("scale reordered alloc");
        {
            let (src_ptr, _src_guard) = scale_checkpoint.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = scale.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_cutlass_int4_reorder_scale_sm90a_cuda(
                    src_ptr as *const crate::ffi::Half,
                    dst_ptr as *mut crate::ffi::Half,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("scale reorder");
        }
        {
            let (src_ptr, _src_guard) = packed_offset_binary.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = packed_reordered.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_cutlass_int4_reorder_weight_sm90a_cuda(
                    src_ptr as *const u8,
                    dst_ptr as *mut u8,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("weight reorder");
        }

        let expert_indptr = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
        let mut output = ctx
            .stream
            .alloc_zeros::<bf16>(routed_tokens * out_dim)
            .expect("output alloc");
        let mut sizes = crate::ffi::KimiCutlassInt4GroupedWorkspaceSizes::default();
        let result = unsafe {
            crate::ffi::kimi_cutlass_int4_grouped_workspace_sizes_sm90a_cuda(
                routed_tokens as i32,
                in_dim as i32,
                out_dim as i32,
                local_experts as i32,
                group_size as i32,
                &raw mut sizes,
            )
        };
        result.result().expect("workspace sizes");
        let mut workspace = ctx
            .stream
            .alloc_zeros::<u8>(sizes.total_bytes)
            .expect("workspace alloc");

        let params = {
            let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = packed_reordered.device_ptr(&ctx.stream);
            let (scale_ptr, _scale_guard) = scale.device_ptr(&ctx.stream);
            let (indptr_ptr, _indptr_guard) = expert_indptr.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
            let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
            let params = crate::ffi::KimiCutlassInt4GroupedLaunchParams {
                input: input_ptr as *const crate::ffi::Half,
                weight_packed_reordered: weight_ptr as *const u8,
                weight_scale: scale_ptr as *const crate::ffi::Half,
                expert_indptr: indptr_ptr as *const u32,
                output: output_ptr as *mut crate::ffi::Half,
                workspace: workspace_ptr as *mut std::ffi::c_void,
                workspace_bytes: sizes.total_bytes,
                routed_tokens: routed_tokens as i32,
                in_dim: in_dim as i32,
                out_dim: out_dim as i32,
                local_experts: local_experts as i32,
                group_size: group_size as i32,
                sm_count: 0,
            };
            let result = unsafe {
                crate::ffi::kimi_cutlass_int4_grouped_prepare_sm90a_cuda(
                    params,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("prepare");
            params
        };
        let result = unsafe {
            crate::ffi::kimi_cutlass_int4_grouped_launch_sm90a_cuda(params, ctx.stream.cu_stream())
        };
        result.result().expect("launch");

        let got = ctx.stream.clone_dtoh(&output).expect("output D2H");
        ctx.sync().expect("sync");

        let mut max_diff = 0.0f32;
        let mut max_message = String::new();
        for (token, case) in cases.iter().enumerate() {
            for other_case in cases {
                let actual = got[token * out_dim + other_case.row].to_f32();
                let expected = if other_case.row == case.row {
                    let signed = i8::try_from(case.unsigned_nibble).expect("nibble") as f32 - 8.0;
                    bf16::from_f32(signed * scale_value(case.row, case.col / group_size)).to_f32()
                } else {
                    0.0
                };
                let diff = (actual - expected).abs();
                if other_case.row == case.row {
                    let signed = i8::try_from(case.unsigned_nibble).expect("nibble") as f32 - 8.0;
                    let inferred_scale = actual / signed;
                    eprintln!(
                        "probe token={token} row={} col={} group={} unsigned_nibble={} signed={signed} inferred_scale={inferred_scale} expected_scale={}",
                        case.row,
                        case.col,
                        case.col / group_size,
                        case.unsigned_nibble,
                        scale_value(case.row, case.col / group_size)
                    );
                }
                if diff > max_diff {
                    max_diff = diff;
                    max_message = format!(
                        "token={token} row={} col={} unsigned_nibble={} actual={actual} expected={expected}",
                        other_case.row, case.col, case.unsigned_nibble
                    );
                }
            }
        }
        assert!(
            max_diff > 1.0e-3,
            "example69 unexpectedly matched Kimi per32 scale semantics; {max_message} max_diff={max_diff}"
        );
        eprintln!("example69 per32 scale mismatch: {max_message} max_diff={max_diff}");
    }

    #[test]
    #[ignore = "H20-only legacy broad synthetic; not a Kimi group-size-32 correctness gate"]
    fn h20_kimi_cutlass_int4_grouped_projection_legacy_broad_synthetic() {
        use half::bf16;

        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let local_experts = KIMI_K2_LOCAL_EXPERTS;
        let group_size = KIMI_K2_INT4_GROUP_SIZE;
        let in_dim = KIMI_K2_HIDDEN;
        let out_dim = KIMI_K2_EXPERT_INTERMEDIATE;
        let indptr = (0..=local_experts)
            .map(|expert| expert.min(2) as u32)
            .collect::<Vec<_>>();
        let routed_tokens = *indptr.last().expect("indptr") as usize;

        let input_host = (0..routed_tokens * in_dim)
            .map(|idx| {
                let value = ((idx % 67) as f32 - 33.0) * 0.01;
                bf16::from_f32(value)
            })
            .collect::<Vec<_>>();

        let mut packed_host = vec![0x88u8; local_experts * out_dim * (in_dim / 2)];
        let mut scale_host =
            vec![bf16::from_f32(0.03125); local_experts * out_dim * (in_dim / group_size)];
        for expert in 0..2 {
            for row in 0..out_dim {
                for group in 0..(in_dim / group_size) {
                    let value = 0.03125 + 0.001 * ((expert + row + group) % 7) as f32;
                    scale_host[(expert * out_dim + row) * (in_dim / group_size) + group] =
                        bf16::from_f32(value);
                }
                for byte_col in 0..(in_dim / 2) {
                    let col0 = 2 * byte_col;
                    let col1 = col0 + 1;
                    let q0 = ((expert + row + col0) % 16) as i32;
                    let q1 = ((expert * 3 + row + col1) % 16) as i32;
                    packed_host[(expert * out_dim + row) * (in_dim / 2) + byte_col] =
                        (q0 as u8 & 0x0f) | ((q1 as u8 & 0x0f) << 4);
                }
            }
        }

        let input = ctx.stream.clone_htod(&input_host).expect("input H2D");
        let packed_offset_binary = ctx.stream.clone_htod(&packed_host).expect("weight H2D");
        let mut packed_reordered = ctx
            .stream
            .alloc_zeros::<u8>(packed_host.len())
            .expect("reordered alloc");
        let scale_checkpoint = ctx.stream.clone_htod(&scale_host).expect("scale H2D");
        let mut scale = ctx
            .stream
            .alloc_zeros::<bf16>(scale_host.len())
            .expect("scale reordered alloc");
        {
            let (src_ptr, _src_guard) = scale_checkpoint.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = scale.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_cutlass_int4_reorder_scale_sm90a_cuda(
                    src_ptr as *const crate::ffi::Half,
                    dst_ptr as *mut crate::ffi::Half,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("scale reorder");
        }
        let expert_indptr = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
        let mut output = ctx
            .stream
            .alloc_zeros::<bf16>(routed_tokens * out_dim)
            .expect("output alloc");

        {
            let (src_ptr, _src_guard) = packed_offset_binary.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = packed_reordered.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_cutlass_int4_reorder_weight_sm90a_cuda(
                    src_ptr as *const u8,
                    dst_ptr as *mut u8,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("reorder");
        }

        let mut sizes = crate::ffi::KimiCutlassInt4GroupedWorkspaceSizes::default();
        let result = unsafe {
            crate::ffi::kimi_cutlass_int4_grouped_workspace_sizes_sm90a_cuda(
                routed_tokens as i32,
                in_dim as i32,
                out_dim as i32,
                local_experts as i32,
                group_size as i32,
                &raw mut sizes,
            )
        };
        result.result().expect("workspace sizes");
        let mut workspace = ctx
            .stream
            .alloc_zeros::<u8>(sizes.total_bytes)
            .expect("workspace alloc");

        let params = {
            let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = packed_reordered.device_ptr(&ctx.stream);
            let (scale_ptr, _scale_guard) = scale.device_ptr(&ctx.stream);
            let (indptr_ptr, _indptr_guard) = expert_indptr.device_ptr(&ctx.stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
            let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
            let params = crate::ffi::KimiCutlassInt4GroupedLaunchParams {
                input: input_ptr as *const crate::ffi::Half,
                weight_packed_reordered: weight_ptr as *const u8,
                weight_scale: scale_ptr as *const crate::ffi::Half,
                expert_indptr: indptr_ptr as *const u32,
                output: output_ptr as *mut crate::ffi::Half,
                workspace: workspace_ptr as *mut std::ffi::c_void,
                workspace_bytes: sizes.total_bytes,
                routed_tokens: routed_tokens as i32,
                in_dim: in_dim as i32,
                out_dim: out_dim as i32,
                local_experts: local_experts as i32,
                group_size: group_size as i32,
                sm_count: 0,
            };
            let result = unsafe {
                crate::ffi::kimi_cutlass_int4_grouped_prepare_sm90a_cuda(
                    params,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("prepare");
            params
        };
        let result = unsafe {
            crate::ffi::kimi_cutlass_int4_grouped_launch_sm90a_cuda(params, ctx.stream.cu_stream())
        };
        result.result().expect("launch");

        let got = ctx.stream.clone_dtoh(&output).expect("output D2H");
        ctx.sync().expect("sync");

        let signed_nibble = |expert: usize, row: usize, col: usize| -> f32 {
            let byte_col = col / 2;
            let byte = packed_host[(expert * out_dim + row) * (in_dim / 2) + byte_col];
            let unsigned = if col % 2 == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            i8::try_from(unsigned).expect("nibble") as f32 - 8.0
        };
        let mut expected = vec![bf16::ZERO; routed_tokens * out_dim];
        for expert in 0..local_experts {
            let start = indptr[expert] as usize;
            let end = indptr[expert + 1] as usize;
            for token in start..end {
                for row in 0..out_dim {
                    let mut acc = 0.0f32;
                    for col in 0..in_dim {
                        let group = col / group_size;
                        let scale = scale_host
                            [(expert * out_dim + row) * (in_dim / group_size) + group]
                            .to_f32();
                        acc += input_host[token * in_dim + col].to_f32()
                            * signed_nibble(expert, row, col)
                            * scale;
                    }
                    expected[token * out_dim + row] = bf16::from_f32(acc);
                }
            }
        }

        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        let mut max_actual = 0.0f32;
        let mut max_expected = 0.0f32;
        for (idx, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (actual.to_f32() - expected.to_f32()).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = idx;
                max_actual = actual.to_f32();
                max_expected = expected.to_f32();
            }
        }
        assert!(
            max_diff <= 8.0e-2,
            "max_idx={max_idx} actual={max_actual} expected={max_expected} max_diff={max_diff}"
        );
    }

    #[test]
    #[ignore = "H20-only: compares Kimi Marlin WNA16 single-layer routed expert path against vLLM"]
    fn h20_kimi_marlin_wna16_single_layer_matches_vllm_reference() {
        use half::bf16;
        use std::path::PathBuf;

        const TOKENS: usize = 4;
        const BLOCK_SIZE: usize = 8;
        let reference_dir = std::env::var("PEGAINFER_KIMI_MARLIN_WNA16_REFERENCE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/kimi_marlin_wna16_reference"));
        let w13_ref_path = reference_dir.join("w13_out_bf16.bin");
        let route_ref_path = reference_dir.join("route_output_bf16.bin");
        let final_ref_path = reference_dir.join("final_bf16.bin");

        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let route_elems = TOKENS * KIMI_K2_TOPK;

        let topk_host = (0..route_elems)
            .map(|idx| {
                let token = idx / KIMI_K2_TOPK;
                let route = idx % KIMI_K2_TOPK;
                i32::try_from((token * 13 + route * 5) % KIMI_K2_LOCAL_EXPERTS).unwrap()
            })
            .collect::<Vec<_>>();
        let denom = (KIMI_K2_TOPK * (KIMI_K2_TOPK + 1) / 2) as f32;
        let topk_weight_host = (0..route_elems)
            .map(|idx| ((idx % KIMI_K2_TOPK) + 1) as f32 / denom)
            .collect::<Vec<_>>();
        let hidden_host = deterministic_bf16(TOKENS * KIMI_K2_HIDDEN, 23, 1.0 / 32.0, -11.0);
        let w13_weight_host = deterministic_qweight_bytes(
            KIMI_K2_LOCAL_EXPERTS * (KIMI_K2_HIDDEN / 16) * (2 * KIMI_K2_EXPERT_INTERMEDIATE * 2),
        );
        let w2_weight_host = deterministic_qweight_bytes(
            KIMI_K2_LOCAL_EXPERTS * (KIMI_K2_EXPERT_INTERMEDIATE / 16) * (KIMI_K2_HIDDEN * 2),
        );
        let w13_scale_host = deterministic_marlin_scale_bf16(
            KIMI_K2_LOCAL_EXPERTS,
            KIMI_K2_HIDDEN / KIMI_K2_INT4_GROUP_SIZE,
            2 * KIMI_K2_EXPERT_INTERMEDIATE,
            17,
            1.0 / 64.0,
            1.0,
        );
        let w2_scale_host = deterministic_marlin_scale_bf16(
            KIMI_K2_LOCAL_EXPERTS,
            KIMI_K2_EXPERT_INTERMEDIATE / KIMI_K2_INT4_GROUP_SIZE,
            KIMI_K2_HIDDEN,
            19,
            1.0 / 64.0,
            1.0,
        );

        let topk_dev = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
        let topk_weight_dev = ctx
            .stream
            .clone_htod(&topk_weight_host)
            .expect("topk weight H2D");
        let hidden_data = ctx.stream.clone_htod(&hidden_host).expect("hidden H2D");
        let w13_weight_dev = ctx.stream.clone_htod(&w13_weight_host).expect("w13 H2D");
        let w2_weight_dev = ctx.stream.clone_htod(&w2_weight_host).expect("w2 H2D");
        let w13_scale_dev = ctx
            .stream
            .clone_htod(&w13_scale_host)
            .expect("w13 scale H2D");
        let w2_scale_dev = ctx.stream.clone_htod(&w2_scale_host).expect("w2 scale H2D");

        let mut route_workspace =
            KimiMarlinRouteWorkspace::new(&ctx, TOKENS, BLOCK_SIZE).expect("route workspace");
        let routing = kimi_moe_marlin_align_block_size(
            &ctx,
            &mut route_workspace,
            &topk_dev,
            TOKENS,
            TOKENS,
            0,
        )
        .expect("route alignment");
        let mut gemm_workspace =
            KimiMarlinWna16Workspace::new(&ctx, routing.max_m_blocks, KIMI_K2_HIDDEN, BLOCK_SIZE)
                .expect("gemm workspace");

        let hidden = crate::tensor::HiddenStates {
            data: hidden_data,
            hidden_dim: KIMI_K2_HIDDEN,
            seq_len: TOKENS,
        };
        let w13_weight = KimiMarlinFusedW13Int4Weight {
            local_experts: KIMI_K2_LOCAL_EXPERTS,
            in_dim: KIMI_K2_HIDDEN,
            intermediate_dim: KIMI_K2_EXPERT_INTERMEDIATE,
            group_size: KIMI_K2_INT4_GROUP_SIZE,
            weight_packed_uint4b8: &w13_weight_dev,
            weight_scale_permuted: &w13_scale_dev,
        };
        let w2_manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W2Down,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );
        let w2_weight = KimiMarlinInt4Weight {
            manifest: w2_manifest,
            weight_packed_uint4b8: &w2_weight_dev,
            weight_scale_permuted: &w2_scale_dev,
        };

        let mut w13_out =
            crate::tensor::HiddenStates::zeros(&ctx, 2 * KIMI_K2_EXPERT_INTERMEDIATE, route_elems)
                .expect("w13 out");
        kimi_marlin_wna16_w13_gemm(
            &ctx,
            &mut gemm_workspace,
            &routing,
            &hidden,
            &w13_weight,
            &topk_weight_dev,
            &mut w13_out,
        )
        .expect("w13 gemm");
        let w13_got = ctx.stream.clone_dtoh(&w13_out.data).expect("w13 D2H");
        let w13_ref = read_bf16_file(&w13_ref_path, route_elems * 2 * KIMI_K2_EXPERT_INTERMEDIATE);
        assert_bf16_close("w13_out", &w13_got, &w13_ref, 0.5, 0.03);

        let mut activated =
            crate::tensor::HiddenStates::zeros(&ctx, KIMI_K2_EXPERT_INTERMEDIATE, route_elems)
                .expect("activated");
        kimi_marlin_w13_swiglu(&ctx, &w13_out, &mut activated).expect("swiglu");

        let mut route_output =
            crate::tensor::HiddenStates::zeros(&ctx, KIMI_K2_HIDDEN, route_elems)
                .expect("route output");
        kimi_marlin_wna16_w2_gemm(
            &ctx,
            &mut gemm_workspace,
            &routing,
            &activated,
            &w2_weight,
            &topk_weight_dev,
            &mut route_output,
        )
        .expect("w2 gemm");
        let route_got = ctx
            .stream
            .clone_dtoh(&route_output.data)
            .expect("route output D2H");
        let route_ref = read_bf16_file(&route_ref_path, route_elems * KIMI_K2_HIDDEN);
        assert_bf16_close("route_output", &route_got, &route_ref, 16.0, 0.03);

        let mut final_out = ctx
            .stream
            .alloc_zeros::<f32>(TOKENS * KIMI_K2_HIDDEN)
            .expect("final out");
        kimi_marlin_sum_topk_rows_f32(&ctx, &route_output, TOKENS, &mut final_out)
            .expect("sum topk");
        let final_got_f32 = ctx.stream.clone_dtoh(&final_out).expect("final D2H");
        let final_got = final_got_f32
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let final_ref = read_bf16_file(&final_ref_path, TOKENS * KIMI_K2_HIDDEN);
        assert_bf16_close("final", &final_got, &final_ref, 128.0, 0.25);
    }

    fn deterministic_qweight_bytes(words: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(words * std::mem::size_of::<u32>());
        for idx in 0..words {
            let word = ((idx as u64 * 1_103_515_245 + 12_345 + (idx as u64 / 97) * 17)
                & 0x7fff_ffff) as u32;
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    fn deterministic_bf16(len: usize, modulo: usize, scale: f32, offset: f32) -> Vec<bf16> {
        (0..len)
            .map(|idx| bf16::from_f32(((idx % modulo) as f32 + offset) * scale))
            .collect()
    }

    fn deterministic_marlin_scale_bf16(
        local_experts: usize,
        groups: usize,
        out_dim: usize,
        modulo: usize,
        scale: f32,
        offset: f32,
    ) -> Vec<bf16> {
        let elements_per_expert = groups * out_dim;
        assert_eq!(elements_per_expert % 64, 0);
        let marlin_scale_perm = |offset: usize| -> usize { offset / 8 + 8 * (offset % 8) };
        let mut out = Vec::with_capacity(local_experts * elements_per_expert);
        for expert in 0..local_experts {
            let expert_base = expert * elements_per_expert;
            for flat in 0..elements_per_expert {
                let source_flat = (flat / 64) * 64 + marlin_scale_perm(flat % 64);
                let raw_idx = expert_base + source_flat;
                out.push(bf16::from_f32(((raw_idx % modulo) as f32 + offset) * scale));
            }
        }
        out
    }

    fn read_bf16_file(path: &std::path::Path, expected: usize) -> Vec<bf16> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        assert_eq!(
            bytes.len(),
            expected * std::mem::size_of::<u16>(),
            "{} len mismatch",
            path.display()
        );
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()
    }

    fn assert_bf16_close(
        name: &str,
        got: &[bf16],
        expected: &[bf16],
        max_limit: f32,
        mean_limit: f32,
    ) {
        assert_eq!(got.len(), expected.len(), "{name} len mismatch");
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut max_idx = 0usize;
        let mut max_got = 0.0f32;
        let mut max_expected = 0.0f32;
        for (idx, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
            let actual = actual.to_f32();
            let expected = expected.to_f32();
            let diff = (actual - expected).abs();
            sum_diff += diff;
            if diff > max_diff {
                max_diff = diff;
                max_idx = idx;
                max_got = actual;
                max_expected = expected;
            }
        }
        let mean_diff = sum_diff / got.len() as f32;
        println!(
            "{name}: max_diff={max_diff} mean_diff={mean_diff} max_idx={max_idx} got={max_got} expected={max_expected}"
        );
        assert!(
            max_diff <= max_limit && mean_diff <= mean_limit,
            "{name} diff too large: max_diff={max_diff} mean_diff={mean_diff} limits=({max_limit}, {mean_limit}) max_idx={max_idx} got={max_got} expected={max_expected}"
        );
    }
}
