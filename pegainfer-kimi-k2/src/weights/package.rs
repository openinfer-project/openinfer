use super::*;

impl KimiGpuRawTensor {
    pub(crate) fn copy_bf16_matrix(
        &self,
        ctx: &KimiRankGpuContext,
        rows: usize,
        cols: usize,
        role: &str,
    ) -> Result<DeviceMatrix> {
        validate_raw_tensor(self, Dtype::BF16, &[rows, cols], role)?;
        Ok(DeviceMatrix {
            data: copy_raw_tensor_to_typed::<bf16>(ctx, self)?,
            rows,
            cols,
        })
    }

    pub(crate) fn copy_bf16_matrix_from_shape(
        &self,
        ctx: &KimiRankGpuContext,
        role: &str,
    ) -> Result<DeviceMatrix> {
        ensure!(
            self.shape.len() == 2,
            "Kimi {role} tensor {} must be rank-2, got {:?}",
            self.name,
            self.shape
        );
        self.copy_bf16_matrix(ctx, self.shape[0], self.shape[1], role)
    }

    pub(crate) fn copy_bf16_vec(
        &self,
        ctx: &KimiRankGpuContext,
        len: usize,
        role: &str,
    ) -> Result<DeviceVec> {
        validate_raw_tensor(self, Dtype::BF16, &[len], role)?;
        Ok(DeviceVec {
            data: copy_raw_tensor_to_typed::<bf16>(ctx, self)?,
            len,
        })
    }
}

impl KimiRankGpuWeights {
    pub fn typed_view<'a>(
        &'a self,
        names: &'a KimiRankWeightNames,
    ) -> Result<KimiRankTypedGpuWeights<'a>> {
        ensure!(
            self.rank == names.rank,
            "Kimi GPU rank {} does not match typed names rank {}",
            self.rank,
            names.rank
        );
        validate_rank_tensor_catalog(names, self.tensors.len(), |name, dtype| {
            self.expect_tensor(name, dtype).map(|_| ())
        })?;

        let top = KimiTopGpuWeights {
            token_embedding: self.expect_tensor(&names.top.token_embedding, Dtype::BF16)?,
            final_norm: self.expect_tensor(&names.top.final_norm, Dtype::BF16)?,
            lm_head: self.expect_tensor(&names.top.lm_head, Dtype::BF16)?,
        };
        let mut layers = Vec::with_capacity(names.layers.len());
        for layer in &names.layers {
            let attention = KimiAttentionGpuWeights {
                input_layernorm: self
                    .expect_tensor(&layer.attention.input_layernorm, Dtype::BF16)?,
                q_a_proj: self.expect_tensor(&layer.attention.q_a_proj, Dtype::BF16)?,
                q_a_layernorm: self.expect_tensor(&layer.attention.q_a_layernorm, Dtype::BF16)?,
                q_b_proj: self.expect_tensor(&layer.attention.q_b_proj, Dtype::BF16)?,
                kv_a_proj_with_mqa: self
                    .expect_tensor(&layer.attention.kv_a_proj_with_mqa, Dtype::BF16)?,
                kv_a_layernorm: self.expect_tensor(&layer.attention.kv_a_layernorm, Dtype::BF16)?,
                kv_b_proj: self.expect_tensor(&layer.attention.kv_b_proj, Dtype::BF16)?,
                o_proj: self.expect_tensor(&layer.attention.o_proj, Dtype::BF16)?,
                post_attention_layernorm: self
                    .expect_tensor(&layer.attention.post_attention_layernorm, Dtype::BF16)?,
            };
            let kind = match &layer.kind {
                KimiLayerWeightKindNames::Dense(mlp) => {
                    KimiLayerKindGpuWeights::Dense(KimiDenseMlpGpuWeights {
                        gate_proj: self.expect_tensor(&mlp.gate_proj, Dtype::BF16)?,
                        up_proj: self.expect_tensor(&mlp.up_proj, Dtype::BF16)?,
                        down_proj: self.expect_tensor(&mlp.down_proj, Dtype::BF16)?,
                    })
                }
                KimiLayerWeightKindNames::Moe(moe) => {
                    let routed_experts = moe
                        .routed_experts
                        .iter()
                        .map(|expert| self.routed_expert_view(expert))
                        .collect::<Result<Vec<_>>>()?;
                    KimiLayerKindGpuWeights::Moe(KimiMoeLayerGpuWeights {
                        router: KimiRouterGpuWeights {
                            gate_weight: self
                                .expect_tensor(&moe.router.gate_weight, Dtype::BF16)?,
                            e_score_correction_bias: self
                                .expect_tensor(&moe.router.e_score_correction_bias, Dtype::F32)?,
                        },
                        shared_experts: KimiSharedExpertGpuWeights {
                            gate_proj: self
                                .expect_tensor(&moe.shared_experts.gate_proj, Dtype::BF16)?,
                            up_proj: self
                                .expect_tensor(&moe.shared_experts.up_proj, Dtype::BF16)?,
                            down_proj: self
                                .expect_tensor(&moe.shared_experts.down_proj, Dtype::BF16)?,
                        },
                        routed_experts,
                    })
                }
            };
            layers.push(KimiLayerGpuWeights {
                layer_idx: layer.layer_idx,
                attention,
                kind,
            });
        }

        Ok(KimiRankTypedGpuWeights {
            rank: self.rank,
            plan: &names.plan,
            top,
            layers,
        })
    }

    fn routed_expert_view<'a>(
        &'a self,
        expert: &'a KimiRoutedExpertWeightNames,
    ) -> Result<KimiRoutedExpertGpuWeights<'a>> {
        Ok(KimiRoutedExpertGpuWeights {
            global_expert: expert.global_expert,
            gate_proj: self.int4_projection_view(&expert.gate_proj)?,
            up_proj: self.int4_projection_view(&expert.up_proj)?,
            down_proj: self.int4_projection_view(&expert.down_proj)?,
        })
    }

    fn int4_projection_view<'a>(
        &'a self,
        projection: &'a KimiInt4ProjectionWeightNames,
    ) -> Result<KimiInt4ProjectionGpuWeights<'a>> {
        Ok(KimiInt4ProjectionGpuWeights {
            weight_packed: self.expect_tensor(&projection.weight_packed, Dtype::I32)?,
            weight_scale: self.expect_tensor(&projection.weight_scale, Dtype::BF16)?,
            weight_shape: self.expect_tensor(&projection.weight_shape, Dtype::I32)?,
        })
    }

    pub fn pack_rank_expert_marlin_weights(
        &mut self,
        ctx: &KimiRankGpuContext,
        names: &KimiRankWeightNames,
    ) -> Result<KimiRankExpertMarlinWeights> {
        ensure!(
            self.rank == names.rank,
            "Kimi GPU rank {} does not match typed names rank {}",
            self.rank,
            names.rank
        );
        ctx.set_current()?;
        let mut layers = Vec::with_capacity(KIMI_K2_MOE_LAYERS);
        let mut packaged_moes = Vec::with_capacity(KIMI_K2_MOE_LAYERS);
        for layer in &names.layers {
            let KimiLayerWeightKindNames::Moe(moe) = &layer.kind else {
                continue;
            };
            validate_local_expert_name_order(
                names.rank,
                layer.layer_idx,
                names.plan.local_expert_range.clone(),
                &moe.routed_experts,
            )?;

            let gate = self.pack_projection_marlin_buffers_from_names(
                ctx,
                names.plan.ep_rank,
                KimiInt4ProjectionRole::Gate,
                moe.routed_experts
                    .iter()
                    .map(|expert| &expert.gate_proj)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
            let up = self.pack_projection_marlin_buffers_from_names(
                ctx,
                names.plan.ep_rank,
                KimiInt4ProjectionRole::Up,
                moe.routed_experts
                    .iter()
                    .map(|expert| &expert.up_proj)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
            let down = self.pack_projection_marlin_buffers_from_names(
                ctx,
                names.plan.ep_rank,
                KimiInt4ProjectionRole::Down,
                moe.routed_experts
                    .iter()
                    .map(|expert| &expert.down_proj)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
            let raw_source_bytes =
                gate.plan.total_bytes() + up.plan.total_bytes() + down.plan.total_bytes();
            let w13 = fuse_expert_major_w13_marlin_buffers(ctx, &gate, &up)?;
            let total_bytes = w13.package_bytes() + down.package_bytes();
            let weights = KimiMoeLayerExpertMarlinWeights {
                layer_idx: layer.layer_idx,
                first_global_expert: names.plan.local_expert_range.start,
                local_experts: names.plan.local_expert_range.len(),
                w13,
                down,
                raw_source_bytes,
                total_bytes,
            };
            weights.as_marlin_weights().validate()?;
            layers.push(weights);
            packaged_moes.push(moe);
        }
        ensure!(
            layers.len() == KIMI_K2_MOE_LAYERS,
            "Kimi rank {} expected {KIMI_K2_MOE_LAYERS} MoE Marlin weight layers, got {}",
            self.rank,
            layers.len()
        );
        let raw_source_bytes = layers.iter().map(|layer| layer.raw_source_bytes).sum();
        let total_bytes = layers.iter().map(|layer| layer.total_bytes).sum();
        self.remove_packaged_routed_expert_raw_tensors(&packaged_moes)?;
        Ok(KimiRankExpertMarlinWeights {
            rank: self.rank,
            local_expert_range: names.plan.local_expert_range.clone(),
            layers,
            raw_source_bytes,
            total_bytes,
        })
    }

    fn pack_projection_marlin_buffers_from_names(
        &self,
        ctx: &KimiRankGpuContext,
        ep_rank: usize,
        role: KimiInt4ProjectionRole,
        projection_names: &[&KimiInt4ProjectionWeightNames],
    ) -> Result<KimiExpertMajorProjectionMarlinBuffers> {
        let projections = projection_names
            .iter()
            .map(|projection| self.int4_projection_view(projection))
            .collect::<Result<Vec<_>>>()?;
        pack_expert_major_projection_marlin_buffers(ctx, ep_rank, role, projections.iter())
    }

    fn remove_packaged_routed_expert_raw_tensors(
        &mut self,
        moes: &[&KimiMoeLayerWeightNames],
    ) -> Result<()> {
        let mut names = Vec::new();
        for moe in moes {
            for expert in &moe.routed_experts {
                push_int4_projection_raw_tensor_names(&expert.gate_proj, &mut names);
                push_int4_projection_raw_tensor_names(&expert.up_proj, &mut names);
                push_int4_projection_raw_tensor_names(&expert.down_proj, &mut names);
            }
        }

        let mut removed_bytes = 0usize;
        for name in &names {
            let tensor = self.tensors.get(name.as_str()).with_context(|| {
                format!("missing Kimi raw tensor {name} during package cleanup")
            })?;
            removed_bytes += tensor.bytes;
        }
        ensure!(
            removed_bytes <= self.total_bytes,
            "Kimi rank {} package cleanup would remove {} bytes from {} total bytes",
            self.rank,
            removed_bytes,
            self.total_bytes
        );

        for name in names {
            let tensor = self
                .tensors
                .remove(name.as_str())
                .expect("validated Kimi raw tensor must exist during package cleanup");
            self.total_bytes -= tensor.bytes;
        }

        Ok(())
    }

    fn expect_tensor(&self, name: &str, dtype: Dtype) -> Result<&KimiGpuRawTensor> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("missing Kimi GPU tensor {name}"))?;
        ensure!(
            tensor.dtype == dtype,
            "Kimi GPU tensor {name} dtype {:?} does not match expected {:?}",
            tensor.dtype,
            dtype
        );
        Ok(tensor)
    }
}

impl KimiRankTypedGpuWeights<'_> {
    pub fn expert_major_weight_plan(&self) -> Result<KimiRankExpertMajorWeightPlan> {
        let mut layers = Vec::with_capacity(KIMI_K2_MOE_LAYERS);
        for layer in &self.layers {
            let KimiLayerKindGpuWeights::Moe(moe) = &layer.kind else {
                continue;
            };
            ensure!(
                moe.routed_experts.len() == self.plan.local_expert_range.len(),
                "Kimi rank {} layer {} expected {} local routed experts, got {}",
                self.rank,
                layer.layer_idx,
                self.plan.local_expert_range.len(),
                moe.routed_experts.len()
            );
            for (offset, expert) in moe.routed_experts.iter().enumerate() {
                let expected = self.plan.local_expert_range.start + offset;
                ensure!(
                    expert.global_expert == expected,
                    "Kimi rank {} layer {} local expert offset {} expected global expert {}, got {}",
                    self.rank,
                    layer.layer_idx,
                    offset,
                    expected,
                    expert.global_expert
                );
            }
            let gate = validate_expert_major_projection(
                KimiInt4ProjectionRole::Gate,
                moe.routed_experts.iter().map(|expert| &expert.gate_proj),
            )
            .with_context(|| {
                format!(
                    "failed to validate Kimi rank {} layer {} gate expert-major weights",
                    self.rank, layer.layer_idx
                )
            })?;
            let up = validate_expert_major_projection(
                KimiInt4ProjectionRole::Up,
                moe.routed_experts.iter().map(|expert| &expert.up_proj),
            )
            .with_context(|| {
                format!(
                    "failed to validate Kimi rank {} layer {} up expert-major weights",
                    self.rank, layer.layer_idx
                )
            })?;
            let down = validate_expert_major_projection(
                KimiInt4ProjectionRole::Down,
                moe.routed_experts.iter().map(|expert| &expert.down_proj),
            )
            .with_context(|| {
                format!(
                    "failed to validate Kimi rank {} layer {} down expert-major weights",
                    self.rank, layer.layer_idx
                )
            })?;
            let total_bytes = gate.total_bytes() + up.total_bytes() + down.total_bytes();
            layers.push(KimiMoeLayerExpertMajorPlan {
                layer_idx: layer.layer_idx,
                first_global_expert: self.plan.local_expert_range.start,
                local_experts: self.plan.local_expert_range.len(),
                gate,
                up,
                down,
                total_bytes,
            });
        }
        ensure!(
            layers.len() == KIMI_K2_MOE_LAYERS,
            "Kimi rank {} expected {KIMI_K2_MOE_LAYERS} MoE layers, got {}",
            self.rank,
            layers.len()
        );
        let total_bytes = layers.iter().map(|layer| layer.total_bytes).sum();
        Ok(KimiRankExpertMajorWeightPlan {
            rank: self.rank,
            local_expert_range: self.plan.local_expert_range.clone(),
            layers,
            total_bytes,
        })
    }

    pub fn pack_expert_major_layer_marlin_weights(
        &self,
        ctx: &KimiRankGpuContext,
        layer_idx: usize,
    ) -> Result<KimiMoeLayerExpertMarlinWeights> {
        ctx.set_current()?;
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.layer_idx == layer_idx)
            .with_context(|| format!("missing Kimi rank {} layer {layer_idx}", self.rank))?;
        let KimiLayerKindGpuWeights::Moe(moe) = &layer.kind else {
            bail!(
                "Kimi rank {} layer {layer_idx} is dense, not MoE",
                self.rank
            );
        };
        validate_local_expert_order(
            self.rank,
            layer.layer_idx,
            self.plan.local_expert_range.clone(),
            &moe.routed_experts,
        )?;

        let gate = pack_expert_major_projection_marlin_buffers(
            ctx,
            self.plan.ep_rank,
            KimiInt4ProjectionRole::Gate,
            moe.routed_experts.iter().map(|expert| &expert.gate_proj),
        )?;
        let up = pack_expert_major_projection_marlin_buffers(
            ctx,
            self.plan.ep_rank,
            KimiInt4ProjectionRole::Up,
            moe.routed_experts.iter().map(|expert| &expert.up_proj),
        )?;
        let w13 = fuse_expert_major_w13_marlin_buffers(ctx, &gate, &up)?;
        let down = pack_expert_major_projection_marlin_buffers(
            ctx,
            self.plan.ep_rank,
            KimiInt4ProjectionRole::Down,
            moe.routed_experts.iter().map(|expert| &expert.down_proj),
        )?;
        let raw_source_bytes =
            gate.plan.total_bytes() + up.plan.total_bytes() + down.plan.total_bytes();
        let total_bytes = w13.package_bytes() + down.package_bytes();
        let weights = KimiMoeLayerExpertMarlinWeights {
            layer_idx: layer.layer_idx,
            first_global_expert: self.plan.local_expert_range.start,
            local_experts: self.plan.local_expert_range.len(),
            w13,
            down,
            raw_source_bytes,
            total_bytes,
        };
        weights.as_marlin_weights().validate()?;
        Ok(weights)
    }
}

impl KimiRouterGpuWeights<'_> {
    pub fn copy_to_device_weights(
        &self,
        ctx: &KimiRankGpuContext,
    ) -> Result<KimiRouterDeviceWeights> {
        ctx.set_current()?;
        validate_raw_tensor(
            self.gate_weight,
            Dtype::BF16,
            &[KIMI_K2_ROUTED_EXPERTS, KIMI_K2_HIDDEN],
            "router gate_weight",
        )?;
        validate_raw_tensor(
            self.e_score_correction_bias,
            Dtype::F32,
            &[KIMI_K2_ROUTED_EXPERTS],
            "router e_score_correction_bias",
        )?;
        let gate_data = copy_raw_tensor_to_typed::<bf16>(ctx, self.gate_weight)?;
        let e_score_correction_bias =
            copy_raw_tensor_to_typed::<f32>(ctx, self.e_score_correction_bias)?;
        Ok(KimiRouterDeviceWeights {
            gate_weight: DeviceMatrix {
                data: gate_data,
                rows: KIMI_K2_ROUTED_EXPERTS,
                cols: KIMI_K2_HIDDEN,
            },
            e_score_correction_bias,
        })
    }
}

impl KimiExpertMajorProjectionPlan {
    fn total_bytes(&self) -> usize {
        self.packed_bytes + self.scale_bytes + self.shape_bytes
    }
}

fn validate_expert_major_projection<'a>(
    role: KimiInt4ProjectionRole,
    projections: impl IntoIterator<Item = &'a KimiInt4ProjectionGpuWeights<'a>>,
) -> Result<KimiExpertMajorProjectionPlan> {
    let projections = projections.into_iter().collect::<Vec<_>>();
    ensure!(
        !projections.is_empty(),
        "Kimi expert-major projection cannot be empty"
    );
    let (out_dim, in_dim) = role.dims();
    let packed_i32_shape = [out_dim, in_dim / 8];
    let scale_bf16_shape = [out_dim, in_dim / KIMI_K2_INT4_GROUP_SIZE];
    let shape_i32_shape = [2];
    let mut packed_bytes = 0usize;
    let mut scale_bytes = 0usize;
    let mut shape_bytes = 0usize;
    for projection in &projections {
        validate_raw_tensor(
            projection.weight_packed,
            Dtype::I32,
            &packed_i32_shape,
            "weight_packed",
        )?;
        validate_raw_tensor(
            projection.weight_scale,
            Dtype::BF16,
            &scale_bf16_shape,
            "weight_scale",
        )?;
        validate_raw_tensor(
            projection.weight_shape,
            Dtype::I32,
            &shape_i32_shape,
            "weight_shape",
        )?;
        packed_bytes += projection.weight_packed.bytes;
        scale_bytes += projection.weight_scale.bytes;
        shape_bytes += projection.weight_shape.bytes;
    }
    Ok(KimiExpertMajorProjectionPlan {
        role,
        local_experts: projections.len(),
        out_dim,
        in_dim,
        packed_i32_shape_per_expert: packed_i32_shape,
        scale_bf16_shape_per_expert: scale_bf16_shape,
        packed_bytes,
        scale_bytes,
        shape_bytes,
    })
}

fn pack_expert_major_projection_marlin_buffers<'a>(
    ctx: &KimiRankGpuContext,
    ep_rank: usize,
    role: KimiInt4ProjectionRole,
    projections: impl IntoIterator<Item = &'a KimiInt4ProjectionGpuWeights<'a>>,
) -> Result<KimiExpertMajorProjectionMarlinBuffers> {
    let projections = projections.into_iter().collect::<Vec<_>>();
    let plan = validate_expert_major_projection(role, projections.iter().copied())?;
    let manifest = KimiInt4WeightManifest::ep8(
        role.kernel_role(),
        ep_rank,
        KimiInt4NibbleOrder::LowThenHigh,
    );
    manifest.validate()?;
    ensure!(
        manifest.local_experts == plan.local_experts
            && manifest.logical_shape.out_dim == plan.out_dim
            && manifest.logical_shape.in_dim == plan.in_dim
            && manifest.packed_shape.elements() == plan.packed_bytes
            && manifest.scale_shape.elements() * std::mem::size_of::<bf16>() == plan.scale_bytes,
        "Kimi {:?} expert-major plan does not match Marlin manifest {:?}",
        role,
        manifest
    );

    let mut weight_packed_offset_binary = ctx
        .stream
        .alloc_zeros::<u8>(manifest.packed_shape.elements())?;
    let mut weight_packed_marlin_uint4b8 = ctx
        .stream
        .alloc_zeros::<u8>(manifest.packed_shape.elements())?;
    let mut weight_scale_checkpoint = ctx
        .stream
        .alloc_zeros::<bf16>(manifest.scale_shape.elements())?;
    let mut weight_scale_marlin_permuted = ctx
        .stream
        .alloc_zeros::<bf16>(manifest.scale_shape.elements())?;

    copy_projection_component_to_contiguous(
        ctx,
        projections
            .iter()
            .map(|projection| projection.weight_packed),
        &mut weight_packed_offset_binary,
        plan.packed_bytes,
        "weight_packed",
    )?;
    kimi_marlin_int4_reorder_weight(
        &ctx.as_device_context(),
        &weight_packed_offset_binary,
        &mut weight_packed_marlin_uint4b8,
        &manifest,
    )?;
    copy_projection_component_to_typed_contiguous(
        ctx,
        projections.iter().map(|projection| projection.weight_scale),
        &mut weight_scale_checkpoint,
        plan.scale_bytes,
        "weight_scale",
    )?;
    kimi_marlin_int4_reorder_scale(
        &ctx.as_device_context(),
        &weight_scale_checkpoint,
        &mut weight_scale_marlin_permuted,
        &manifest,
    )?;

    Ok(KimiExpertMajorProjectionMarlinBuffers {
        role,
        plan,
        manifest,
        weight_packed_marlin_uint4b8,
        weight_scale_marlin_permuted,
    })
}

fn fuse_expert_major_w13_marlin_buffers(
    ctx: &KimiRankGpuContext,
    gate: &KimiExpertMajorProjectionMarlinBuffers,
    up: &KimiExpertMajorProjectionMarlinBuffers,
) -> Result<KimiExpertMajorW13MarlinBuffers> {
    ensure!(
        gate.role == KimiInt4ProjectionRole::Gate && up.role == KimiInt4ProjectionRole::Up,
        "Kimi Marlin W13 fuse expects gate/up roles, got {:?}/{:?}",
        gate.role,
        up.role
    );
    ensure!(
        gate.plan.local_experts == up.plan.local_experts
            && gate.plan.in_dim == up.plan.in_dim
            && gate.plan.out_dim == up.plan.out_dim
            && gate.plan.out_dim == KIMI_K2_EXPERT_INTERMEDIATE
            && gate.plan.in_dim == KIMI_K2_HIDDEN,
        "Kimi Marlin W13 fuse shape mismatch: gate {:?}, up {:?}",
        gate.plan,
        up.plan
    );
    let mut weight_packed_marlin_uint4b8 = ctx.stream.alloc_zeros::<u8>(
        gate.weight_packed_marlin_uint4b8.len() + up.weight_packed_marlin_uint4b8.len(),
    )?;
    let mut weight_scale_marlin_permuted = ctx.stream.alloc_zeros::<bf16>(
        gate.weight_scale_marlin_permuted.len() + up.weight_scale_marlin_permuted.len(),
    )?;

    kimi_marlin_int4_fuse_w13(
        &ctx.as_device_context(),
        &gate.as_marlin_weight(),
        &up.as_marlin_weight(),
        &mut weight_packed_marlin_uint4b8,
        &mut weight_scale_marlin_permuted,
    )?;

    let fused = KimiExpertMajorW13MarlinBuffers {
        local_experts: gate.plan.local_experts,
        in_dim: gate.plan.in_dim,
        intermediate_dim: gate.plan.out_dim,
        group_size: KIMI_K2_INT4_GROUP_SIZE,
        weight_packed_marlin_uint4b8,
        weight_scale_marlin_permuted,
    };
    fused.as_marlin_weight().validate()?;
    Ok(fused)
}

fn copy_projection_component_to_contiguous<'a>(
    ctx: &KimiRankGpuContext,
    tensors: impl IntoIterator<Item = &'a KimiGpuRawTensor>,
    dst: &mut CudaSlice<u8>,
    expected_bytes: usize,
    component: &str,
) -> Result<()> {
    ensure!(
        dst.len() == expected_bytes,
        "Kimi expert-major {component} destination length {} does not match expected {}",
        dst.len(),
        expected_bytes
    );
    let mut offset = 0usize;
    for tensor in tensors {
        let end = offset + tensor.bytes;
        ensure!(
            end <= expected_bytes,
            "Kimi expert-major {component} copy would exceed destination: end {end}, expected {expected_bytes}"
        );
        ctx.stream
            .memcpy_dtod(
                &tensor.data.slice(0..tensor.bytes),
                &mut dst.slice_mut(offset..end),
            )
            .with_context(|| {
                format!(
                    "failed to D2D copy Kimi expert-major {component} tensor {}",
                    tensor.name
                )
            })?;
        offset = end;
    }
    ensure!(
        offset == expected_bytes,
        "Kimi expert-major {component} copied {offset} bytes, expected {expected_bytes}"
    );
    Ok(())
}

fn copy_raw_tensor_to_typed<T: DeviceRepr + ValidAsZeroBits>(
    ctx: &KimiRankGpuContext,
    tensor: &KimiGpuRawTensor,
) -> Result<CudaSlice<T>> {
    let element_bytes = std::mem::size_of::<T>();
    ensure!(
        tensor.bytes.is_multiple_of(element_bytes),
        "Kimi tensor {} byte size {} is not divisible by typed element size {}",
        tensor.name,
        tensor.bytes,
        element_bytes
    );
    let mut dst = ctx.stream.alloc_zeros::<T>(tensor.bytes / element_bytes)?;
    {
        let (src_ptr, _src_guard) = tensor.data.device_ptr(&ctx.stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&ctx.stream);
        unsafe {
            cuda_result::memcpy_dtod_async(dst_ptr, src_ptr, tensor.bytes, ctx.stream.cu_stream())
        }
        .with_context(|| {
            format!(
                "failed to D2D copy Kimi tensor {} into typed GPU buffer",
                tensor.name
            )
        })?;
    }
    Ok(dst)
}

fn copy_projection_component_to_typed_contiguous<'a, T: DeviceRepr>(
    ctx: &KimiRankGpuContext,
    tensors: impl IntoIterator<Item = &'a KimiGpuRawTensor>,
    dst: &mut CudaSlice<T>,
    expected_bytes: usize,
    component: &str,
) -> Result<()> {
    let dst_bytes = dst.len() * std::mem::size_of::<T>();
    ensure!(
        dst_bytes == expected_bytes,
        "Kimi expert-major {component} destination bytes {dst_bytes} does not match expected {expected_bytes}"
    );
    let mut offset = 0usize;
    for tensor in tensors {
        let end = offset + tensor.bytes;
        ensure!(
            end <= expected_bytes,
            "Kimi expert-major {component} copy would exceed destination: end {end}, expected {expected_bytes}"
        );
        let (src_ptr, _src_guard) = tensor.data.device_ptr(&ctx.stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&ctx.stream);
        // SAFETY: this is a byte-preserving D2D copy from the raw safetensors
        // GPU payload into a typed buffer with the same total byte count. The
        // dtype and shape were validated immediately before allocation.
        unsafe {
            cuda_result::memcpy_dtod_async(
                dst_ptr + offset as u64,
                src_ptr,
                tensor.bytes,
                ctx.stream.cu_stream(),
            )
        }
        .with_context(|| {
            format!(
                "failed to D2D copy Kimi expert-major {component} tensor {} into typed package",
                tensor.name
            )
        })?;
        offset = end;
    }
    ensure!(
        offset == expected_bytes,
        "Kimi expert-major {component} copied {offset} bytes, expected {expected_bytes}"
    );
    Ok(())
}

fn validate_local_expert_order(
    rank: usize,
    layer_idx: usize,
    local_expert_range: Range<usize>,
    routed_experts: &[KimiRoutedExpertGpuWeights<'_>],
) -> Result<()> {
    ensure!(
        routed_experts.len() == local_expert_range.len(),
        "Kimi rank {} layer {} expected {} local routed experts, got {}",
        rank,
        layer_idx,
        local_expert_range.len(),
        routed_experts.len()
    );
    for (offset, expert) in routed_experts.iter().enumerate() {
        let expected = local_expert_range.start + offset;
        ensure!(
            expert.global_expert == expected,
            "Kimi rank {} layer {} local expert offset {} expected global expert {}, got {}",
            rank,
            layer_idx,
            offset,
            expected,
            expert.global_expert
        );
    }
    Ok(())
}

fn validate_local_expert_name_order(
    rank: usize,
    layer_idx: usize,
    local_expert_range: Range<usize>,
    routed_experts: &[KimiRoutedExpertWeightNames],
) -> Result<()> {
    ensure!(
        routed_experts.len() == local_expert_range.len(),
        "Kimi rank {} layer {} expected {} local routed expert names, got {}",
        rank,
        layer_idx,
        local_expert_range.len(),
        routed_experts.len()
    );
    for (offset, expert) in routed_experts.iter().enumerate() {
        let expected = local_expert_range.start + offset;
        ensure!(
            expert.global_expert == expected,
            "Kimi rank {} layer {} local expert name offset {} expected global expert {}, got {}",
            rank,
            layer_idx,
            offset,
            expected,
            expert.global_expert
        );
    }
    Ok(())
}

fn push_int4_projection_raw_tensor_names(
    projection: &KimiInt4ProjectionWeightNames,
    out: &mut Vec<String>,
) {
    out.push(projection.weight_packed.clone());
    out.push(projection.weight_scale.clone());
    out.push(projection.weight_shape.clone());
}

fn validate_raw_tensor(
    tensor: &KimiGpuRawTensor,
    dtype: Dtype,
    shape: &[usize],
    role: &str,
) -> Result<()> {
    ensure!(
        tensor.dtype == dtype,
        "Kimi {role} tensor {} dtype {:?} does not match expected {:?}",
        tensor.name,
        tensor.dtype,
        dtype
    );
    ensure!(
        tensor.shape == shape,
        "Kimi {role} tensor {} shape {:?} does not match expected {:?}",
        tensor.name,
        tensor.shape,
        shape
    );
    let expected_bytes = shape.iter().product::<usize>() * dtype_element_bytes(dtype)?;
    ensure!(
        tensor.bytes == expected_bytes,
        "Kimi {role} tensor {} bytes {} does not match expected {}",
        tensor.name,
        tensor.bytes,
        expected_bytes
    );
    Ok(())
}
