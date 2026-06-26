use std::{collections::BTreeSet, ops::Range};

use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, result as cuda_result};
use safetensors::Dtype;

use super::load::Glm52GpuRawTensor;
use super::view::{
    Glm52Fp8ProjectionWeightNames, Glm52LayerWeightKindNames, Glm52MoeLayerWeightNames,
    Glm52RankWeightNames, Glm52RoutedExpertWeightNames,
};
use super::{Glm52RankGpuContext, Glm52RankGpuWeights, expected_tensor_contract};
use crate::config::{
    GLM52_DENSE_LAYERS, GLM52_EXPERT_INTERMEDIATE, GLM52_HIDDEN, GLM52_MOE_LAYERS,
};

const GLM52_FP8_BLOCK_SIZE: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52Fp8ExpertProjectionRole {
    Gate,
    Up,
    Down,
}

impl Glm52Fp8ExpertProjectionRole {
    const fn dims(self) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (GLM52_EXPERT_INTERMEDIATE, GLM52_HIDDEN),
            Self::Down => (GLM52_HIDDEN, GLM52_EXPERT_INTERMEDIATE),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Glm52Fp8ExpertScaleLayout {
    CheckpointBlock128x128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DeepGemmMGroupedFp8WeightPlan {
    pub(crate) groups: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) block_size: usize,
    pub(crate) scale_rows: usize,
    pub(crate) scale_cols: usize,
    pub(crate) weight_elems: usize,
    pub(crate) scale_elems: usize,
}

impl Glm52DeepGemmMGroupedFp8WeightPlan {
    fn new(groups: usize, n: usize, k: usize) -> Result<Self> {
        ensure!(
            groups > 0 && n > 0 && k > 0,
            "GLM5.2 DeepGEMM m-grouped FP8 plan needs nonzero dimensions: groups={groups}, n={n}, k={k}"
        );
        ensure!(
            n.is_multiple_of(GLM52_FP8_BLOCK_SIZE) && k.is_multiple_of(GLM52_FP8_BLOCK_SIZE),
            "GLM5.2 DeepGEMM m-grouped FP8 plan expects 128-aligned N/K, got n={n}, k={k}"
        );
        let scale_rows = n / GLM52_FP8_BLOCK_SIZE;
        let scale_cols = k / GLM52_FP8_BLOCK_SIZE;
        Ok(Self {
            groups,
            n,
            k,
            block_size: GLM52_FP8_BLOCK_SIZE,
            scale_rows,
            scale_cols,
            weight_elems: groups * n * k,
            scale_elems: groups * scale_rows * scale_cols,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52ExpertMajorProjectionPlan {
    pub(crate) local_experts: usize,
    pub(crate) out_dim: usize,
    pub(crate) in_dim: usize,
    pub(crate) weight_bytes: usize,
    pub(crate) scale_rows: usize,
    pub(crate) scale_cols: usize,
    pub(crate) scale_bytes: usize,
    pub(crate) scale_layout: Glm52Fp8ExpertScaleLayout,
}

pub(crate) struct Glm52ExpertMajorProjectionFp8Buffers {
    pub(crate) role: Glm52Fp8ExpertProjectionRole,
    pub(crate) plan: Glm52ExpertMajorProjectionPlan,
    pub(crate) weight_e4m3: CudaSlice<u8>,
    pub(crate) weight_scale_inv_f32: CudaSlice<f32>,
}

impl Glm52ExpertMajorProjectionFp8Buffers {
    fn package_bytes(&self) -> usize {
        self.weight_e4m3.len() + self.weight_scale_inv_f32.len() * std::mem::size_of::<f32>()
    }

    pub(crate) fn deepgemm_m_grouped_plan(&self) -> Result<Glm52DeepGemmMGroupedFp8WeightPlan> {
        ensure!(
            self.plan.scale_layout == Glm52Fp8ExpertScaleLayout::CheckpointBlock128x128,
            "GLM5.2 {:?} expert package scale layout {:?} is not DeepGEMM FP8 block-128 compatible",
            self.role,
            self.plan.scale_layout
        );
        let plan = Glm52DeepGemmMGroupedFp8WeightPlan::new(
            self.plan.local_experts,
            self.plan.out_dim,
            self.plan.in_dim,
        )?;
        ensure!(
            self.weight_e4m3.len() == plan.weight_elems
                && self.weight_scale_inv_f32.len() == plan.scale_elems
                && self.plan.scale_rows == plan.scale_rows
                && self.plan.scale_cols == plan.scale_cols,
            "GLM5.2 {:?} expert package does not match DeepGEMM [G,N,K] / [G,N/128,K/128] contract: package weight={}, scale={}, rows={}, cols={}, plan={plan:?}",
            self.role,
            self.weight_e4m3.len(),
            self.weight_scale_inv_f32.len(),
            self.plan.scale_rows,
            self.plan.scale_cols
        );
        Ok(plan)
    }
}

pub(crate) struct Glm52ExpertMajorW13Fp8Buffers {
    pub(crate) local_experts: usize,
    pub(crate) in_dim: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) block_size: usize,
    pub(crate) scale_layout: Glm52Fp8ExpertScaleLayout,
    pub(crate) weight_e4m3: CudaSlice<u8>,
    pub(crate) weight_scale_inv_f32: CudaSlice<f32>,
}

impl Glm52ExpertMajorW13Fp8Buffers {
    fn package_bytes(&self) -> usize {
        self.weight_e4m3.len() + self.weight_scale_inv_f32.len() * std::mem::size_of::<f32>()
    }

    pub(crate) fn deepgemm_m_grouped_plan(&self) -> Result<Glm52DeepGemmMGroupedFp8WeightPlan> {
        ensure!(
            self.scale_layout == Glm52Fp8ExpertScaleLayout::CheckpointBlock128x128,
            "GLM5.2 W13 expert package scale layout {:?} is not DeepGEMM FP8 block-128 compatible",
            self.scale_layout
        );
        let plan = Glm52DeepGemmMGroupedFp8WeightPlan::new(
            self.local_experts,
            2 * self.intermediate_dim,
            self.in_dim,
        )?;
        ensure!(
            self.weight_e4m3.len() == plan.weight_elems
                && self.weight_scale_inv_f32.len() == plan.scale_elems,
            "GLM5.2 W13 expert package does not match DeepGEMM [G,N,K] / [G,N/128,K/128] contract: package weight={}, scale={}, plan={plan:?}",
            self.weight_e4m3.len(),
            self.weight_scale_inv_f32.len()
        );
        Ok(plan)
    }
}

pub(crate) struct Glm52MoeLayerExpertFp8Weights {
    pub(crate) layer_idx: usize,
    pub(crate) w13: Glm52ExpertMajorW13Fp8Buffers,
    pub(crate) down: Glm52ExpertMajorProjectionFp8Buffers,
    pub(crate) total_bytes: usize,
}

pub(crate) struct Glm52RankExpertFp8Weights {
    pub(crate) rank: usize,
    pub(crate) local_expert_range: Range<usize>,
    pub(crate) layers: Vec<Glm52MoeLayerExpertFp8Weights>,
    pub(crate) total_bytes: usize,
}

struct Glm52Fp8ProjectionRaw<'a> {
    weight: &'a Glm52GpuRawTensor,
    weight_scale_inv: &'a Glm52GpuRawTensor,
}

impl Glm52RankGpuWeights {
    pub(crate) fn pack_loaded_expert_fp8_layers(
        &mut self,
        ctx: &Glm52RankGpuContext,
        names: &Glm52RankWeightNames,
        packed_layers: &mut BTreeSet<usize>,
        out: &mut Vec<Glm52MoeLayerExpertFp8Weights>,
    ) -> Result<()> {
        ensure!(
            self.rank == names.rank,
            "GLM5.2 GPU rank {} does not match typed names rank {}",
            self.rank,
            names.rank
        );
        ctx.set_current()?;
        for layer in &names.layers {
            let Glm52LayerWeightKindNames::Moe(moe) = &layer.kind else {
                continue;
            };
            if packed_layers.contains(&layer.layer_idx) || !self.has_all_routed_expert_raw(moe) {
                continue;
            }
            let weights =
                self.pack_moe_layer_expert_fp8_weights(ctx, names, layer.layer_idx, moe)?;
            self.remove_packaged_routed_expert_raw_tensors(&[moe])?;
            packed_layers.insert(layer.layer_idx);
            out.push(weights);
        }
        Ok(())
    }

    fn pack_moe_layer_expert_fp8_weights(
        &self,
        ctx: &Glm52RankGpuContext,
        names: &Glm52RankWeightNames,
        layer_idx: usize,
        moe: &Glm52MoeLayerWeightNames,
    ) -> Result<Glm52MoeLayerExpertFp8Weights> {
        validate_local_expert_name_order(
            names.rank,
            layer_idx,
            names.plan.local_expert_range.clone(),
            &moe.routed_experts,
        )?;

        let w13 = self.pack_w13_fp8_buffers_from_experts(ctx, &moe.routed_experts)?;
        let down = self.pack_projection_fp8_buffers_from_names(
            ctx,
            Glm52Fp8ExpertProjectionRole::Down,
            &moe.routed_experts
                .iter()
                .map(|expert| &expert.down_proj)
                .collect::<Vec<_>>(),
        )?;
        let total_bytes = w13.package_bytes() + down.package_bytes();
        Ok(Glm52MoeLayerExpertFp8Weights {
            layer_idx,
            w13,
            down,
            total_bytes,
        })
    }

    fn pack_projection_fp8_buffers_from_names(
        &self,
        ctx: &Glm52RankGpuContext,
        role: Glm52Fp8ExpertProjectionRole,
        projection_names: &[&Glm52Fp8ProjectionWeightNames],
    ) -> Result<Glm52ExpertMajorProjectionFp8Buffers> {
        let projections = projection_names
            .iter()
            .map(|projection| self.fp8_projection_raw(projection))
            .collect::<Result<Vec<_>>>()?;
        pack_expert_major_projection_fp8_buffers(ctx, role, projections.iter())
    }

    fn pack_w13_fp8_buffers_from_experts(
        &self,
        ctx: &Glm52RankGpuContext,
        experts: &[Glm52RoutedExpertWeightNames],
    ) -> Result<Glm52ExpertMajorW13Fp8Buffers> {
        let gate = experts
            .iter()
            .map(|expert| self.fp8_projection_raw(&expert.gate_proj))
            .collect::<Result<Vec<_>>>()?;
        let up = experts
            .iter()
            .map(|expert| self.fp8_projection_raw(&expert.up_proj))
            .collect::<Result<Vec<_>>>()?;
        pack_expert_major_w13_fp8_buffers(ctx, &gate, &up)
    }

    fn fp8_projection_raw<'a>(
        &'a self,
        names: &'a Glm52Fp8ProjectionWeightNames,
    ) -> Result<Glm52Fp8ProjectionRaw<'a>> {
        Ok(Glm52Fp8ProjectionRaw {
            weight: expect_resident_tensor(&self.tensors, &names.weight)?,
            weight_scale_inv: expect_resident_tensor(&self.tensors, &names.weight_scale_inv)?,
        })
    }

    fn has_all_routed_expert_raw(&self, moe: &Glm52MoeLayerWeightNames) -> bool {
        moe.routed_experts.iter().all(|expert| {
            has_fp8_projection_raw(&self.tensors, &expert.gate_proj)
                && has_fp8_projection_raw(&self.tensors, &expert.up_proj)
                && has_fp8_projection_raw(&self.tensors, &expert.down_proj)
        })
    }

    fn remove_packaged_routed_expert_raw_tensors(
        &mut self,
        moes: &[&Glm52MoeLayerWeightNames],
    ) -> Result<()> {
        let mut names = Vec::new();
        for moe in moes {
            for expert in &moe.routed_experts {
                push_fp8_projection_raw_tensor_names(&expert.gate_proj, &mut names);
                push_fp8_projection_raw_tensor_names(&expert.up_proj, &mut names);
                push_fp8_projection_raw_tensor_names(&expert.down_proj, &mut names);
            }
        }

        let mut removed_bytes = 0usize;
        for name in &names {
            let tensor = self.tensors.get(name.as_str()).with_context(|| {
                format!("missing GLM5.2 raw tensor {name} during package cleanup")
            })?;
            removed_bytes += tensor.bytes;
        }
        ensure!(
            removed_bytes <= self.total_bytes,
            "GLM5.2 rank {} package cleanup would remove {} bytes from {} total bytes",
            self.rank,
            removed_bytes,
            self.total_bytes
        );

        for name in names {
            let tensor = self
                .tensors
                .remove(name.as_str())
                .expect("validated GLM5.2 raw tensor must exist during package cleanup");
            self.total_bytes -= tensor.bytes;
        }
        Ok(())
    }
}

fn pack_expert_major_projection_fp8_buffers<'a>(
    ctx: &Glm52RankGpuContext,
    role: Glm52Fp8ExpertProjectionRole,
    projections: impl IntoIterator<Item = &'a Glm52Fp8ProjectionRaw<'a>>,
) -> Result<Glm52ExpertMajorProjectionFp8Buffers> {
    let projections = projections.into_iter().collect::<Vec<_>>();
    let plan = validate_expert_major_projection(role, projections.iter().copied())?;
    let mut weight_e4m3 = ctx.stream().alloc_zeros::<u8>(plan.weight_bytes)?;
    let mut weight_scale_inv_f32 = ctx
        .stream()
        .alloc_zeros::<f32>(plan.scale_bytes / std::mem::size_of::<f32>())?;

    copy_projection_component_to_contiguous(
        ctx,
        projections.iter().map(|projection| projection.weight),
        &mut weight_e4m3,
        plan.weight_bytes,
        "weight",
    )?;
    copy_projection_component_to_typed_contiguous(
        ctx,
        projections
            .iter()
            .map(|projection| projection.weight_scale_inv),
        &mut weight_scale_inv_f32,
        plan.scale_bytes,
        "weight_scale_inv",
    )?;

    Ok(Glm52ExpertMajorProjectionFp8Buffers {
        role,
        plan,
        weight_e4m3,
        weight_scale_inv_f32,
    })
}

fn validate_expert_major_projection<'a>(
    role: Glm52Fp8ExpertProjectionRole,
    projections: impl IntoIterator<Item = &'a Glm52Fp8ProjectionRaw<'a>>,
) -> Result<Glm52ExpertMajorProjectionPlan> {
    let projections = projections.into_iter().collect::<Vec<_>>();
    ensure!(
        !projections.is_empty(),
        "GLM5.2 expert-major FP8 projection cannot be empty"
    );
    let (out_dim, in_dim) = role.dims();
    let weight_shape = [out_dim, in_dim];
    let scale_rows = out_dim.div_ceil(GLM52_FP8_BLOCK_SIZE);
    let scale_cols = in_dim.div_ceil(GLM52_FP8_BLOCK_SIZE);
    let scale_shape = [scale_rows, scale_cols];
    let mut weight_bytes = 0usize;
    let mut scale_bytes = 0usize;
    for projection in &projections {
        validate_raw_tensor(projection.weight, Dtype::F8_E4M3, &weight_shape, "weight")?;
        validate_raw_tensor(
            projection.weight_scale_inv,
            Dtype::F32,
            &scale_shape,
            "weight_scale_inv",
        )?;
        weight_bytes += projection.weight.bytes;
        scale_bytes += projection.weight_scale_inv.bytes;
    }
    Ok(Glm52ExpertMajorProjectionPlan {
        local_experts: projections.len(),
        out_dim,
        in_dim,
        weight_bytes,
        scale_rows,
        scale_cols,
        scale_bytes,
        scale_layout: Glm52Fp8ExpertScaleLayout::CheckpointBlock128x128,
    })
}

fn pack_expert_major_w13_fp8_buffers(
    ctx: &Glm52RankGpuContext,
    gate: &[Glm52Fp8ProjectionRaw<'_>],
    up: &[Glm52Fp8ProjectionRaw<'_>],
) -> Result<Glm52ExpertMajorW13Fp8Buffers> {
    let gate_plan =
        validate_expert_major_projection(Glm52Fp8ExpertProjectionRole::Gate, gate.iter())?;
    let up_plan = validate_expert_major_projection(Glm52Fp8ExpertProjectionRole::Up, up.iter())?;
    ensure!(
        gate.len() == up.len(),
        "GLM5.2 FP8 W13 package expects matching gate/up local expert counts, got {}/{}",
        gate.len(),
        up.len()
    );
    ensure!(
        gate_plan.local_experts == up_plan.local_experts
            && gate_plan.in_dim == up_plan.in_dim
            && gate_plan.out_dim == up_plan.out_dim
            && gate_plan.scale_rows == up_plan.scale_rows
            && gate_plan.scale_cols == up_plan.scale_cols
            && gate_plan.scale_layout == up_plan.scale_layout
            && gate_plan.out_dim == GLM52_EXPERT_INTERMEDIATE
            && gate_plan.in_dim == GLM52_HIDDEN,
        "GLM5.2 FP8 W13 package shape mismatch: gate {:?}, up {:?}",
        gate_plan,
        up_plan
    );

    let mut weight_e4m3 = ctx
        .stream()
        .alloc_zeros::<u8>(gate_plan.weight_bytes + up_plan.weight_bytes)?;
    let mut weight_scale_inv_f32 = ctx.stream().alloc_zeros::<f32>(
        (gate_plan.scale_bytes + up_plan.scale_bytes) / std::mem::size_of::<f32>(),
    )?;

    let mut ordered_weights = Vec::with_capacity(gate.len() * 2);
    let mut ordered_scales = Vec::with_capacity(gate.len() * 2);
    for (gate, up) in gate.iter().zip(up) {
        ordered_weights.push(gate.weight);
        ordered_weights.push(up.weight);
        ordered_scales.push(gate.weight_scale_inv);
        ordered_scales.push(up.weight_scale_inv);
    }
    copy_projection_component_to_contiguous(
        ctx,
        ordered_weights,
        &mut weight_e4m3,
        gate_plan.weight_bytes + up_plan.weight_bytes,
        "w13 weight",
    )?;
    copy_projection_component_to_typed_contiguous(
        ctx,
        ordered_scales,
        &mut weight_scale_inv_f32,
        gate_plan.scale_bytes + up_plan.scale_bytes,
        "w13 weight_scale_inv",
    )?;

    Ok(Glm52ExpertMajorW13Fp8Buffers {
        local_experts: gate_plan.local_experts,
        in_dim: gate_plan.in_dim,
        intermediate_dim: gate_plan.out_dim,
        block_size: GLM52_FP8_BLOCK_SIZE,
        scale_layout: gate_plan.scale_layout,
        weight_e4m3,
        weight_scale_inv_f32,
    })
}

fn copy_projection_component_to_contiguous<'a>(
    ctx: &Glm52RankGpuContext,
    tensors: impl IntoIterator<Item = &'a Glm52GpuRawTensor>,
    dst: &mut CudaSlice<u8>,
    expected_bytes: usize,
    component: &str,
) -> Result<()> {
    ensure!(
        dst.len() == expected_bytes,
        "GLM5.2 expert-major {component} destination length {} does not match expected {}",
        dst.len(),
        expected_bytes
    );
    let mut offset = 0usize;
    for tensor in tensors {
        let end = offset + tensor.bytes;
        ensure!(
            end <= expected_bytes,
            "GLM5.2 expert-major {component} copy would exceed destination: end {end}, expected {expected_bytes}"
        );
        ctx.stream()
            .memcpy_dtod(
                &tensor.data.slice(0..tensor.bytes),
                &mut dst.slice_mut(offset..end),
            )
            .with_context(|| {
                format!(
                    "failed to D2D copy GLM5.2 expert-major {component} tensor {}",
                    tensor.name
                )
            })?;
        offset = end;
    }
    ensure!(
        offset == expected_bytes,
        "GLM5.2 expert-major {component} copied {offset} bytes, expected {expected_bytes}"
    );
    Ok(())
}

fn copy_projection_component_to_typed_contiguous<'a, T: DeviceRepr>(
    ctx: &Glm52RankGpuContext,
    tensors: impl IntoIterator<Item = &'a Glm52GpuRawTensor>,
    dst: &mut CudaSlice<T>,
    expected_bytes: usize,
    component: &str,
) -> Result<()> {
    let dst_bytes = dst.len() * std::mem::size_of::<T>();
    ensure!(
        dst_bytes == expected_bytes,
        "GLM5.2 expert-major {component} destination bytes {dst_bytes} does not match expected {expected_bytes}"
    );
    let mut offset = 0usize;
    for tensor in tensors {
        let end = offset + tensor.bytes;
        ensure!(
            end <= expected_bytes,
            "GLM5.2 expert-major {component} copy would exceed destination: end {end}, expected {expected_bytes}"
        );
        let (src_ptr, _src_guard) = tensor.data.device_ptr(ctx.stream());
        let (dst_ptr, _dst_guard) = dst.device_ptr_mut(ctx.stream());
        unsafe {
            cuda_result::memcpy_dtod_async(
                dst_ptr + offset as u64,
                src_ptr,
                tensor.bytes,
                ctx.stream().cu_stream(),
            )
        }
        .with_context(|| {
            format!(
                "failed to D2D copy GLM5.2 expert-major {component} tensor {} into typed package",
                tensor.name
            )
        })?;
        offset = end;
    }
    ensure!(
        offset == expected_bytes,
        "GLM5.2 expert-major {component} copied {offset} bytes, expected {expected_bytes}"
    );
    Ok(())
}

fn expect_resident_tensor<'a>(
    tensors: &'a std::collections::BTreeMap<String, Glm52GpuRawTensor>,
    name: &str,
) -> Result<&'a Glm52GpuRawTensor> {
    let tensor = tensors
        .get(name)
        .with_context(|| format!("missing GLM5.2 GPU tensor {name}"))?;
    let contract = expected_tensor_contract(name)?;
    ensure!(
        tensor.dtype == contract.dtype && tensor.shape == contract.shape,
        "GLM5.2 GPU tensor {name} contract changed after load"
    );
    Ok(tensor)
}

fn has_fp8_projection_raw(
    tensors: &std::collections::BTreeMap<String, Glm52GpuRawTensor>,
    projection: &Glm52Fp8ProjectionWeightNames,
) -> bool {
    tensors.contains_key(&projection.weight) && tensors.contains_key(&projection.weight_scale_inv)
}

fn push_fp8_projection_raw_tensor_names(
    projection: &Glm52Fp8ProjectionWeightNames,
    out: &mut Vec<String>,
) {
    out.push(projection.weight.clone());
    out.push(projection.weight_scale_inv.clone());
}

fn validate_raw_tensor(
    tensor: &Glm52GpuRawTensor,
    dtype: Dtype,
    shape: &[usize],
    role: &str,
) -> Result<()> {
    ensure!(
        tensor.dtype == dtype,
        "GLM5.2 {role} tensor {} dtype {:?} does not match expected {:?}",
        tensor.name,
        tensor.dtype,
        dtype
    );
    ensure!(
        tensor.shape == shape,
        "GLM5.2 {role} tensor {} shape {:?} does not match expected {:?}",
        tensor.name,
        tensor.shape,
        shape
    );
    Ok(())
}

fn validate_local_expert_name_order(
    rank: usize,
    layer_idx: usize,
    local_expert_range: Range<usize>,
    routed_experts: &[Glm52RoutedExpertWeightNames],
) -> Result<()> {
    ensure!(
        routed_experts.len() == local_expert_range.len(),
        "GLM5.2 rank {} layer {} expected {} local routed expert names, got {}",
        rank,
        layer_idx,
        local_expert_range.len(),
        routed_experts.len()
    );
    for (offset, expert) in routed_experts.iter().enumerate() {
        let expected = local_expert_range.start + offset;
        ensure!(
            expert.global_expert == expected,
            "GLM5.2 rank {} layer {} local expert name offset {} expected global expert {}, got {}",
            rank,
            layer_idx,
            offset,
            expected,
            expert.global_expert
        );
    }
    Ok(())
}

impl Glm52RankExpertFp8Weights {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.layers.len() == GLM52_MOE_LAYERS,
            "GLM5.2 rank {} expected {GLM52_MOE_LAYERS} FP8 expert packages, got {}",
            self.rank,
            self.layers.len()
        );
        let summed: usize = self.layers.iter().map(|layer| layer.total_bytes).sum();
        ensure!(
            self.total_bytes == summed,
            "GLM5.2 rank {} FP8 expert package bytes {} do not match summed layer bytes {}",
            self.rank,
            self.total_bytes,
            summed
        );
        let local_experts = self.local_expert_range.len();
        let hidden_blocks = GLM52_HIDDEN.div_ceil(GLM52_FP8_BLOCK_SIZE);
        let intermediate_blocks = GLM52_EXPERT_INTERMEDIATE.div_ceil(GLM52_FP8_BLOCK_SIZE);
        let expected_w13_weight_len = local_experts * 2 * GLM52_EXPERT_INTERMEDIATE * GLM52_HIDDEN;
        let expected_w13_scale_len = local_experts * 2 * intermediate_blocks * hidden_blocks;
        let expected_down_weight_len = local_experts * GLM52_HIDDEN * GLM52_EXPERT_INTERMEDIATE;
        let expected_down_scale_len = local_experts * hidden_blocks * intermediate_blocks;
        for (offset, layer) in self.layers.iter().enumerate() {
            let expected_layer_idx = GLM52_DENSE_LAYERS + offset;
            ensure!(
                layer.layer_idx == expected_layer_idx,
                "GLM5.2 rank {} FP8 expert package order drifted at offset {offset}: expected layer {}, got {}",
                self.rank,
                expected_layer_idx,
                layer.layer_idx
            );
            let w13_deepgemm = layer.w13.deepgemm_m_grouped_plan()?;
            let down_deepgemm = layer.down.deepgemm_m_grouped_plan()?;
            ensure!(
                w13_deepgemm
                    == Glm52DeepGemmMGroupedFp8WeightPlan::new(
                        local_experts,
                        2 * GLM52_EXPERT_INTERMEDIATE,
                        GLM52_HIDDEN,
                    )?
                    && down_deepgemm
                        == Glm52DeepGemmMGroupedFp8WeightPlan::new(
                            local_experts,
                            GLM52_HIDDEN,
                            GLM52_EXPERT_INTERMEDIATE,
                        )?,
                "GLM5.2 rank {} layer {} DeepGEMM expert package plan drifted: W13={w13_deepgemm:?}, down={down_deepgemm:?}",
                self.rank,
                layer.layer_idx
            );
            ensure!(
                layer.w13.local_experts == local_experts
                    && layer.down.plan.local_experts == local_experts,
                "GLM5.2 rank {} layer {} local expert count drifted",
                self.rank,
                layer.layer_idx
            );
            ensure!(
                layer.w13.in_dim == GLM52_HIDDEN
                    && layer.w13.intermediate_dim == GLM52_EXPERT_INTERMEDIATE
                    && layer.w13.block_size == GLM52_FP8_BLOCK_SIZE
                    && layer.w13.scale_layout == Glm52Fp8ExpertScaleLayout::CheckpointBlock128x128,
                "GLM5.2 rank {} layer {} W13 package shape drifted",
                self.rank,
                layer.layer_idx
            );
            ensure!(
                layer.w13.weight_e4m3.len() == expected_w13_weight_len
                    && layer.w13.weight_scale_inv_f32.len() == expected_w13_scale_len,
                "GLM5.2 rank {} layer {} W13 package length drifted: weight {}, scale {}, expected {}/{}",
                self.rank,
                layer.layer_idx,
                layer.w13.weight_e4m3.len(),
                layer.w13.weight_scale_inv_f32.len(),
                expected_w13_weight_len,
                expected_w13_scale_len
            );
            ensure!(
                layer.down.role == Glm52Fp8ExpertProjectionRole::Down
                    && layer.down.plan.out_dim == GLM52_HIDDEN
                    && layer.down.plan.in_dim == GLM52_EXPERT_INTERMEDIATE
                    && layer.down.plan.scale_layout
                        == Glm52Fp8ExpertScaleLayout::CheckpointBlock128x128,
                "GLM5.2 rank {} layer {} down package shape drifted",
                self.rank,
                layer.layer_idx
            );
            ensure!(
                layer.down.weight_e4m3.len() == expected_down_weight_len
                    && layer.down.weight_scale_inv_f32.len() == expected_down_scale_len
                    && layer.down.plan.weight_bytes == expected_down_weight_len
                    && layer.down.plan.scale_bytes
                        == expected_down_scale_len * std::mem::size_of::<f32>(),
                "GLM5.2 rank {} layer {} down package length drifted: weight {}, scale {}, plan bytes {}/{}, expected {}/{}/{}",
                self.rank,
                layer.layer_idx,
                layer.down.weight_e4m3.len(),
                layer.down.weight_scale_inv_f32.len(),
                layer.down.plan.weight_bytes,
                layer.down.plan.scale_bytes,
                expected_down_weight_len,
                expected_down_scale_len,
                expected_down_scale_len * std::mem::size_of::<f32>()
            );
        }
        Ok(())
    }
}
