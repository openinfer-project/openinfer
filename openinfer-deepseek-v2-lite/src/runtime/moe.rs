use std::{collections::BTreeMap, env};

use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use openinfer_core::{
    ops,
    tensor::{HiddenStates, HiddenStatesRef},
};
use openinfer_kernels::ops::{
    Dsv2LiteRouterOutput, dsv2_lite_router_softmax_topk_into,
    dsv2_lite_router_softmax_topk_ref_into,
};

use super::{
    DeepSeekV2LiteEp2Generator,
    backend::EpBackendRuntime,
    routing::{MoeRouteEntry, MoeRoutePlan},
};
use crate::{
    attribution::DecodeAttributionProfile,
    device::activate,
    host_ops::{
        gate_logits_host, hidden_from_bf16_host, hidden_from_f32_host, hidden_to_bf16,
        hidden_to_f32, topk_softmax_routes,
    },
    model::{
        DenseMlpForwardScratch, ExpertMlp, MoeMlp, dense_mlp_forward, dense_mlp_forward_per_token,
        dense_mlp_forward_preallocated_into, dense_mlp_forward_preallocated_ref_into,
    },
    nccl_backend::NaiveNcclEp2Backend,
};

#[derive(Clone, Copy)]
pub(super) struct HostStagedRouteWork {
    pub(super) token: usize,
    pub(super) global_expert: usize,
    pub(super) owner_rank: usize,
    pub(super) weight: f32,
}

impl HostStagedRouteWork {
    pub(super) fn new(token: usize, global_expert: usize, owner_rank: usize, weight: f32) -> Self {
        Self {
            token,
            global_expert,
            owner_rank,
            weight,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum HostStagedExpertBatchPolicy {
    Batched,
    Serial,
}

impl HostStagedExpertBatchPolicy {
    pub(super) fn from_env_value(raw: Option<&str>) -> Self {
        match raw.map(str::trim).filter(|value| !value.is_empty()) {
            Some("0" | "false" | "off" | "serial" | "legacy") => Self::Serial,
            _ => Self::Batched,
        }
    }

    fn use_serial(self) -> bool {
        self == Self::Serial
    }
}

fn host_staged_expert_batch_policy() -> HostStagedExpertBatchPolicy {
    HostStagedExpertBatchPolicy::from_env_value(
        env::var("OPENINFER_DSV2_LITE_HOST_STAGED_EXPERT_BATCH")
            .ok()
            .as_deref(),
    )
}

pub(super) fn scatter_host_staged_group_output(
    route_outputs: &mut [Option<Vec<f32>>],
    route_indices: &[usize],
    out: Vec<f32>,
    hidden_size: usize,
) -> Result<()> {
    ensure!(
        hidden_size > 0,
        "host-staged expert output requires nonzero hidden size"
    );
    let expected_len = route_indices
        .len()
        .checked_mul(hidden_size)
        .context("host-staged expert output length overflow")?;
    ensure!(
        out.len() == expected_len,
        "host-staged expert output len mismatch: got {}, expected {}",
        out.len(),
        expected_len
    );
    for (group_row, route_index) in route_indices.iter().copied().enumerate() {
        ensure!(
            route_index < route_outputs.len(),
            "host-staged route index {route_index} exceeds output slots {}",
            route_outputs.len()
        );
        ensure!(
            route_outputs[route_index].is_none(),
            "host-staged route output {route_index} was assigned more than once"
        );
        let begin = group_row * hidden_size;
        let end = begin + hidden_size;
        route_outputs[route_index] = Some(out[begin..end].to_vec());
    }
    Ok(())
}

pub(super) fn accumulate_host_staged_route_output(
    dst: &mut [f32],
    route: HostStagedRouteWork,
    out: Vec<f32>,
    hidden_size: usize,
) -> Result<()> {
    ensure!(
        hidden_size > 0,
        "host-staged route output requires nonzero hidden size"
    );
    ensure!(
        out.len() == hidden_size,
        "host-staged route output len mismatch: got {}, expected {hidden_size}",
        out.len()
    );
    let token_end = route
        .token
        .checked_add(1)
        .and_then(|tokens| tokens.checked_mul(hidden_size))
        .context("host-staged route output offset overflow")?;
    ensure!(
        token_end <= dst.len(),
        "host-staged route token {} exceeds contribution rows {}",
        route.token,
        dst.len() / hidden_size
    );
    let token_begin = token_end - hidden_size;
    for (dst, value) in dst[token_begin..token_end].iter_mut().zip(out) {
        *dst += route.weight * value;
    }
    Ok(())
}

pub(super) struct FixedTopologyMoeScratch {
    rank0_topk_weight: CudaSlice<f32>,
    rank0_topk_idx: CudaSlice<i32>,
    rank1_topk_weight: CudaSlice<f32>,
    rank1_topk_idx: CudaSlice<i32>,
    shared: DenseMlpForwardScratch,
    rank0_expert: DenseMlpForwardScratch,
    rank1_expert: DenseMlpForwardScratch,
    routed: HiddenStates,
}

impl FixedTopologyMoeScratch {
    pub(super) fn new(
        generator: &DeepSeekV2LiteEp2Generator,
        layer_idx: usize,
        moe: &MoeMlp,
        seq_len: usize,
    ) -> Result<Self> {
        let topk_elems = seq_len * generator.config.num_experts_per_token;
        let first_rank0_expert = generator.rank0.layout.owned_experts().start;
        let first_rank1_expert = generator.rank1.layout.owned_experts().start;
        let first_rank0 = generator
            .rank0
            .routed_expert(layer_idx, first_rank0_expert)?;
        let first_rank1 = generator
            .rank1
            .routed_expert(layer_idx, first_rank1_expert)?;
        activate(&generator.rank0.ctx)?;
        let rank0_topk_weight = generator.rank0.ctx.stream.alloc_zeros::<f32>(topk_elems)?;
        let rank0_topk_idx = generator.rank0.ctx.stream.alloc_zeros::<i32>(topk_elems)?;
        let shared = DenseMlpForwardScratch::new(&generator.rank0.ctx, &moe.shared, seq_len)?;
        let rank0_expert =
            DenseMlpForwardScratch::new(&generator.rank0.ctx, &first_rank0.dense, seq_len)?;
        let routed =
            HiddenStates::zeros(&generator.rank0.ctx, generator.config.hidden_size, seq_len)?;
        activate(&generator.rank1.ctx)?;
        let rank1_topk_weight = generator.rank1.ctx.stream.alloc_zeros::<f32>(topk_elems)?;
        let rank1_topk_idx = generator.rank1.ctx.stream.alloc_zeros::<i32>(topk_elems)?;
        let rank1_expert =
            DenseMlpForwardScratch::new(&generator.rank1.ctx, &first_rank1.dense, seq_len)?;
        Ok(Self {
            rank0_topk_weight,
            rank0_topk_idx,
            rank1_topk_weight,
            rank1_topk_idx,
            shared,
            rank0_expert,
            rank1_expert,
            routed,
        })
    }
}

impl DeepSeekV2LiteEp2Generator {
    pub(super) fn moe_forward(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        match &self.backend {
            EpBackendRuntime::HostStaged => self.moe_forward_host_staged(
                layer_idx,
                input,
                moe,
                attribution,
                phase,
                token_index,
                shared_per_token_gemm,
            ),
            EpBackendRuntime::Nccl(nccl) => self.moe_forward_nccl(
                nccl,
                layer_idx,
                input,
                moe,
                attribution,
                phase,
                token_index,
                shared_per_token_gemm,
            ),
        }
    }

    fn moe_forward_host_staged(
        &self,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let (input_host, routes) = attribution.record_result(
            phase,
            "ep_route_host",
            || format!("layer.{layer_idx}.host_staged.route"),
            Some(layer_idx),
            token_index,
            || {
                let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                let route_logits_host = gate_logits_host(&self.config, &input_host, &moe.gate_host);
                let routes = topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);
                Ok((input_host, routes))
            },
        )?;

        let shared = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_expert_enqueue",
            || format!("layer.{layer_idx}.shared_expert"),
            Some(layer_idx),
            token_index,
            || {
                if shared_per_token_gemm {
                    dense_mlp_forward_per_token(&self.rank0.ctx, &moe.shared, input)
                } else {
                    dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)
                }
            },
        )?;
        let route_count = routes.iter().map(Vec::len).sum();
        let mut route_work = Vec::with_capacity(route_count);
        let mut route_groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            for &(global_expert, weight) in token_routes {
                let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
                let route_index = route_work.len();
                route_work.push(HostStagedRouteWork::new(
                    token,
                    global_expert,
                    owner_rank,
                    weight,
                ));
                route_groups
                    .entry((owner_rank, global_expert))
                    .or_default()
                    .push(route_index);
                if owner_rank == 0 {
                    local_routes += 1;
                } else {
                    remote_routes += 1;
                }
            }
        }

        let mut route_outputs = vec![None; route_work.len()];
        if host_staged_expert_batch_policy().use_serial() {
            for (route_index, route) in route_work.iter().enumerate() {
                let section = if route.owner_rank == 0 {
                    "host_staged_local_expert"
                } else {
                    "host_staged_remote_dispatch"
                };
                let expert_ctx = if route.owner_rank == 0 {
                    &self.rank0.ctx
                } else {
                    &self.rank1.ctx
                };
                let begin = route.token * self.config.hidden_size;
                let end = begin + self.config.hidden_size;
                let (out, is_remote) = attribution.record_gpu_result(
                    expert_ctx,
                    phase,
                    section,
                    || format!("layer.{layer_idx}.{section}"),
                    Some(layer_idx),
                    token_index,
                    || {
                        self.expert_forward_host_token(
                            layer_idx,
                            route.global_expert,
                            &input_host[begin..end],
                        )
                    },
                )?;
                let expected_remote = route.owner_rank != 0;
                ensure!(
                    is_remote == expected_remote,
                    "host-staged expert owner mismatch for expert {}: expected remote={expected_remote}, got {is_remote}",
                    route.global_expert
                );
                route_outputs[route_index] = Some(out);
            }
        } else {
            for ((owner_rank, global_expert), route_indices) in route_groups {
                let section = if owner_rank == 0 {
                    "host_staged_local_expert"
                } else {
                    "host_staged_remote_dispatch"
                };
                let expert_ctx = if owner_rank == 0 {
                    &self.rank0.ctx
                } else {
                    &self.rank1.ctx
                };
                let (out, is_remote) = attribution.record_gpu_result(
                    expert_ctx,
                    phase,
                    section,
                    || format!("layer.{layer_idx}.{section}.expert{global_expert}"),
                    Some(layer_idx),
                    token_index,
                    || {
                        self.expert_forward_host_batch(
                            layer_idx,
                            global_expert,
                            &input_host,
                            &route_work,
                            &route_indices,
                        )
                    },
                )?;
                let expected_remote = owner_rank != 0;
                ensure!(
                    is_remote == expected_remote,
                    "host-staged expert owner mismatch for expert {global_expert}: expected remote={expected_remote}, got {is_remote}"
                );
                scatter_host_staged_group_output(
                    &mut route_outputs,
                    &route_indices,
                    out,
                    self.config.hidden_size,
                )?;
            }
        }

        let mut rank0_contrib = vec![0.0f32; input.seq_len * self.config.hidden_size];
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        for (route_index, route) in route_work.iter().enumerate() {
            let dst = if route.owner_rank == 0 {
                &mut rank0_contrib
            } else {
                &mut rank1_contrib
            };
            let out = route_outputs[route_index].take().with_context(|| {
                format!("missing host-staged expert output for route {route_index}")
            })?;
            attribution.record_result(
                phase,
                "host_staged_combine_accumulate",
                || format!("layer.{layer_idx}.host_staged.combine_accumulate"),
                Some(layer_idx),
                token_index,
                || accumulate_host_staged_route_output(dst, *route, out, self.config.hidden_size),
            )?;
        }
        let routed_accum: Vec<_> = rank0_contrib
            .into_iter()
            .zip(rank1_contrib)
            .map(|(rank0, rank1)| rank0 + rank1)
            .collect();

        let routed = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "host_staged_combine_to_device",
            || format!("layer.{layer_idx}.host_staged.combine_to_device"),
            Some(layer_idx),
            token_index,
            || {
                hidden_from_f32_host(
                    &self.rank0.ctx,
                    &routed_accum,
                    self.config.hidden_size,
                    input.seq_len,
                )
            },
        )?;
        activate(&self.rank0.ctx)?;
        let hidden = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_plus_routed_enqueue",
            || format!("layer.{layer_idx}.shared_plus_routed"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &routed, &shared),
        )?;
        Ok((hidden, local_routes, remote_routes))
    }

    fn moe_forward_nccl(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
        shared_per_token_gemm: bool,
    ) -> Result<(HiddenStates, usize, usize)> {
        activate(&self.rank0.ctx)?;
        let route_plan = attribution.record_result(
            phase,
            "ep_route_host",
            || format!("layer.{layer_idx}.nccl.route"),
            Some(layer_idx),
            token_index,
            || {
                let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                let route_logits_host = gate_logits_host(&self.config, &input_host, &moe.gate_host);
                let routes = topk_softmax_routes(&self.config, &route_logits_host, input.seq_len);
                MoeRoutePlan::from_topk_routes(&routes, &self.rank0.layout)
            },
        )?;

        let shared = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_expert_enqueue",
            || format!("layer.{layer_idx}.shared_expert"),
            Some(layer_idx),
            token_index,
            || {
                if shared_per_token_gemm {
                    dense_mlp_forward_per_token(&self.rank0.ctx, &moe.shared, input)
                } else {
                    dense_mlp_forward(&self.rank0.ctx, &moe.shared, input)
                }
            },
        )?;
        let rank1_input = attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_dense_exchange",
            || format!("layer.{layer_idx}.nccl.dense_exchange"),
            Some(layer_idx),
            token_index,
            || nccl.dense_all_reduce_rank0_hidden_to_rank1(&self.rank0.ctx, &self.rank1.ctx, input),
        )?;
        let rank1_hidden = rank1_input.rank1_hidden()?;
        attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_combine_clear",
            || format!("layer.{layer_idx}.nccl.combine_clear"),
            Some(layer_idx),
            token_index,
            || {
                nccl.clear_device_combine(
                    &self.rank0.ctx,
                    &self.rank1.ctx,
                    input.hidden_dim,
                    input.seq_len,
                )
            },
        )?;
        let live_expert_outputs = self.replay_nccl_route_plan(
            nccl,
            layer_idx,
            input,
            rank1_hidden,
            &route_plan,
            attribution,
            phase,
            token_index,
        )?;

        let routed = attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_combine",
            || format!("layer.{layer_idx}.nccl.combine"),
            Some(layer_idx),
            token_index,
            || {
                nccl.combine_device_contributions_to_rank0(
                    &self.rank0.ctx,
                    &self.rank1.ctx,
                    input.hidden_dim,
                    input.seq_len,
                )
            },
        )?;
        drop(live_expert_outputs);
        activate(&self.rank0.ctx)?;
        let hidden = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "shared_plus_routed_enqueue",
            || format!("layer.{layer_idx}.shared_plus_routed"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &routed, &shared),
        )?;
        Ok((
            hidden,
            route_plan.local_routes(),
            route_plan.remote_routes(),
        ))
    }

    pub(super) fn moe_forward_nccl_fixed_topology_preallocated_into(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        moe: &MoeMlp,
        scratch: &mut FixedTopologyMoeScratch,
        out: &mut HiddenStates,
    ) -> Result<()> {
        activate(&self.rank0.ctx)?;
        dsv2_lite_router_softmax_topk_into(
            &self.rank0.ctx,
            input,
            &moe.gate_device,
            self.config.num_experts_per_token,
            &mut Dsv2LiteRouterOutput {
                topk_weight: &mut scratch.rank0_topk_weight,
                topk_idx: &mut scratch.rank0_topk_idx,
            },
        )?;

        dense_mlp_forward_preallocated_into(
            &self.rank0.ctx,
            &moe.shared,
            input,
            &mut scratch.shared,
        )?;

        let rank1_input =
            nccl.dense_all_reduce_rank0_hidden_to_rank1(&self.rank0.ctx, &self.rank1.ctx, input)?;
        let rank1_hidden = rank1_input.rank1_hidden()?;
        activate(&self.rank1.ctx)?;
        dsv2_lite_router_softmax_topk_ref_into(
            &self.rank1.ctx,
            rank1_hidden,
            self.rank1.gate_device(layer_idx)?,
            self.config.num_experts_per_token,
            &mut Dsv2LiteRouterOutput {
                topk_weight: &mut scratch.rank1_topk_weight,
                topk_idx: &mut scratch.rank1_topk_idx,
            },
        )?;

        nccl.clear_device_combine(
            &self.rank0.ctx,
            &self.rank1.ctx,
            input.hidden_dim,
            input.seq_len,
        )?;

        for global_expert in self.rank0.layout.owned_experts() {
            let expert = self.rank0.routed_expert(layer_idx, global_expert)?;
            dense_mlp_forward_preallocated_into(
                &self.rank0.ctx,
                &expert.dense,
                input,
                &mut scratch.rank0_expert,
            )?;
            nccl.accumulate_fixed_expert_contribution(
                0,
                &self.rank0.ctx,
                &scratch.rank0_expert.out,
                &scratch.rank0_topk_weight,
                &scratch.rank0_topk_idx,
                global_expert,
                self.config.num_experts_per_token,
            )?;
        }

        for global_expert in self.rank1.layout.owned_experts() {
            let expert = self.rank1.routed_expert(layer_idx, global_expert)?;
            dense_mlp_forward_preallocated_ref_into(
                &self.rank1.ctx,
                &expert.dense,
                rank1_hidden,
                &mut scratch.rank1_expert,
            )?;
            nccl.accumulate_fixed_expert_contribution(
                1,
                &self.rank1.ctx,
                &scratch.rank1_expert.out,
                &scratch.rank1_topk_weight,
                &scratch.rank1_topk_idx,
                global_expert,
                self.config.num_experts_per_token,
            )?;
        }

        nccl.combine_device_contributions_to_rank0_into(
            &self.rank0.ctx,
            &self.rank1.ctx,
            input.hidden_dim,
            input.seq_len,
            &mut scratch.routed,
        )?;
        drop(rank1_input);
        activate(&self.rank0.ctx)?;
        ops::add_batch_into(&self.rank0.ctx, &scratch.routed, &scratch.shared.out, out)
    }

    fn expert_forward_host_batch(
        &self,
        layer_idx: usize,
        global_expert: usize,
        input_host: &[bf16],
        route_work: &[HostStagedRouteWork],
        route_indices: &[usize],
    ) -> Result<(Vec<f32>, bool)> {
        let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
        let (ctx, expert) = match owner_rank {
            0 => (
                &self.rank0.ctx,
                self.rank0.routed_expert(layer_idx, global_expert)?,
            ),
            1 => (
                &self.rank1.ctx,
                self.rank1.routed_expert(layer_idx, global_expert)?,
            ),
            other => bail!("routed expert {global_expert} maps to unsupported EP rank {other}"),
        };

        ensure!(
            !route_indices.is_empty(),
            "host-staged expert batch requires at least one route"
        );
        let mut batch_input = Vec::with_capacity(route_indices.len() * self.config.hidden_size);
        for &route_index in route_indices {
            let route = route_work[route_index];
            ensure!(
                route.global_expert == global_expert && route.owner_rank == owner_rank,
                "host-staged expert batch mixed route: expected expert {global_expert}/rank {owner_rank}, got expert {}/rank {}",
                route.global_expert,
                route.owner_rank
            );
            let begin = route.token * self.config.hidden_size;
            let end = begin + self.config.hidden_size;
            batch_input.extend_from_slice(&input_host[begin..end]);
        }
        let input = hidden_from_bf16_host(
            ctx,
            &batch_input,
            self.config.hidden_size,
            route_indices.len(),
        )?;
        let out = dense_mlp_forward_per_token(ctx, &expert.dense, &input)?;
        Ok((hidden_to_f32(ctx, &out)?, owner_rank != 0))
    }

    fn expert_forward_host_token(
        &self,
        layer_idx: usize,
        global_expert: usize,
        token_input: &[bf16],
    ) -> Result<(Vec<f32>, bool)> {
        let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
        let (ctx, expert) = match owner_rank {
            0 => (
                &self.rank0.ctx,
                self.rank0.routed_expert(layer_idx, global_expert)?,
            ),
            1 => (
                &self.rank1.ctx,
                self.rank1.routed_expert(layer_idx, global_expert)?,
            ),
            other => bail!("routed expert {global_expert} maps to unsupported EP rank {other}"),
        };

        let input = hidden_from_bf16_host(ctx, token_input, self.config.hidden_size, 1)?;
        let out = dense_mlp_forward(ctx, &expert.dense, &input)?;
        Ok((hidden_to_f32(ctx, &out)?, owner_rank != 0))
    }

    fn replay_nccl_route_plan(
        &self,
        nccl: &NaiveNcclEp2Backend,
        layer_idx: usize,
        input: &HiddenStates,
        rank1_hidden: HiddenStatesRef<'_>,
        route_plan: &MoeRoutePlan,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<Vec<HiddenStates>> {
        let mut live_expert_outputs = Vec::with_capacity(route_plan.route_count());
        for route in route_plan.entries() {
            let out = self.forward_nccl_route(
                layer_idx,
                input.as_ref(),
                rank1_hidden,
                route,
                attribution,
                phase,
                token_index,
            )?;
            let expert_ctx = match route.owner_rank {
                0 => &self.rank0.ctx,
                1 => &self.rank1.ctx,
                other => bail!(
                    "routed expert {} maps to unsupported EP rank {other}",
                    route.global_expert
                ),
            };
            attribution.record_gpu_result(
                expert_ctx,
                phase,
                "nccl_contribution_accumulate_device",
                || format!("layer.{layer_idx}.nccl.contribution_accumulate_device"),
                Some(layer_idx),
                token_index,
                || {
                    nccl.accumulate_device_contribution(
                        route.owner_rank,
                        expert_ctx,
                        &out,
                        route.token,
                        input.seq_len,
                        route.weight,
                    )
                },
            )?;
            live_expert_outputs.push(out);
        }
        Ok(live_expert_outputs)
    }

    fn forward_nccl_route(
        &self,
        layer_idx: usize,
        rank0_hidden: HiddenStatesRef<'_>,
        rank1_hidden: HiddenStatesRef<'_>,
        route: &MoeRouteEntry,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<HiddenStates> {
        match route.owner_rank {
            0 => {
                let expert = self.rank0.routed_expert(layer_idx, route.global_expert)?;
                attribution.record_gpu_result(
                    &self.rank0.ctx,
                    phase,
                    "nccl_local_expert",
                    || format!("layer.{layer_idx}.nccl.local_expert"),
                    Some(layer_idx),
                    token_index,
                    || expert_forward_device(&self.rank0.ctx, expert, rank0_hidden, route.token),
                )
            }
            1 => {
                let expert = self.rank1.routed_expert(layer_idx, route.global_expert)?;
                attribution.record_gpu_result(
                    &self.rank1.ctx,
                    phase,
                    "nccl_remote_expert",
                    || format!("layer.{layer_idx}.nccl.remote_expert"),
                    Some(layer_idx),
                    token_index,
                    || expert_forward_device(&self.rank1.ctx, expert, rank1_hidden, route.token),
                )
            }
            other => bail!(
                "routed expert {} maps to unsupported EP rank {other}",
                route.global_expert
            ),
        }
    }
}

fn expert_forward_device(
    ctx: &openinfer_core::tensor::DeviceContext,
    expert: &ExpertMlp,
    input: HiddenStatesRef<'_>,
    token_idx: usize,
) -> Result<HiddenStates> {
    activate(ctx)?;
    let token = ops::extract_vec_ref(ctx, input, token_idx)?;
    let token_hidden = HiddenStates {
        hidden_dim: token.len,
        seq_len: 1,
        data: token.data,
    };
    dense_mlp_forward(ctx, &expert.dense, &token_hidden)
}

#[cfg(test)]
mod tests;
