use anyhow::{Result, bail};
use half::bf16;
use pegainfer_core::{ops, tensor::HiddenStates};

use super::{DeepSeekV2LiteEp2Generator, backend::EpBackendRuntime};
use crate::{
    attribution::DecodeAttributionProfile,
    device::activate,
    host_ops::{
        gate_logits_host, hidden_from_bf16_host, hidden_from_f32_host, hidden_to_bf16,
        hidden_to_f32, topk_softmax_routes,
    },
    model::{ExpertMlp, MoeMlp, dense_mlp_forward, dense_mlp_forward_per_token},
    nccl_backend::NaiveNcclEp2Backend,
};

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
        let mut rank0_contrib = vec![0.0f32; input.seq_len * self.config.hidden_size];
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            let token_input =
                &input_host[token * self.config.hidden_size..(token + 1) * self.config.hidden_size];
            for &(global_expert, weight) in token_routes {
                let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
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
                let dst = if owner_rank == 0 {
                    &mut rank0_contrib
                } else {
                    &mut rank1_contrib
                };
                let (out, is_remote) = attribution.record_gpu_result(
                    expert_ctx,
                    phase,
                    section,
                    || format!("layer.{layer_idx}.{section}"),
                    Some(layer_idx),
                    token_index,
                    || self.expert_forward_host(layer_idx, global_expert, token_input),
                )?;
                if is_remote {
                    remote_routes += 1;
                } else {
                    local_routes += 1;
                }
                let offset = token * self.config.hidden_size;
                attribution.record_result(
                    phase,
                    "host_staged_combine_accumulate",
                    || format!("layer.{layer_idx}.host_staged.combine_accumulate"),
                    Some(layer_idx),
                    token_index,
                    || {
                        for (dst, value) in dst[offset..offset + self.config.hidden_size]
                            .iter_mut()
                            .zip(out)
                        {
                            *dst += weight * value;
                        }
                        Ok(())
                    },
                )?;
            }
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
        let routes = attribution.record_result(
            phase,
            "ep_route_host",
            || format!("layer.{layer_idx}.nccl.route"),
            Some(layer_idx),
            token_index,
            || {
                let input_host = hidden_to_bf16(&self.rank0.ctx, input)?;
                let route_logits_host = gate_logits_host(&self.config, &input_host, &moe.gate_host);
                Ok(topk_softmax_routes(
                    &self.config,
                    &route_logits_host,
                    input.seq_len,
                ))
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
        let mut rank0_contrib = vec![0.0f32; input.seq_len * self.config.hidden_size];
        let mut rank1_contrib = vec![0.0f32; rank0_contrib.len()];
        // NCCL covers only the dense hidden exchange and final contribution
        // sum in this first gate. Route iteration and expert-output
        // accumulation stay host-side so host-staged remains a simple oracle.
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
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            for &(global_expert, weight) in token_routes {
                let owner_rank = self.rank0.layout.owner_rank(global_expert)?;
                let (out, dst) = match owner_rank {
                    0 => {
                        local_routes += 1;
                        let expert = self.rank0.routed_expert(layer_idx, global_expert)?;
                        (
                            attribution.record_gpu_result(
                                &self.rank0.ctx,
                                phase,
                                "nccl_local_expert",
                                || format!("layer.{layer_idx}.nccl.local_expert"),
                                Some(layer_idx),
                                token_index,
                                || expert_forward_device(&self.rank0.ctx, expert, input, token),
                            )?,
                            &mut rank0_contrib,
                        )
                    }
                    1 => {
                        remote_routes += 1;
                        let expert = self.rank1.routed_expert(layer_idx, global_expert)?;
                        (
                            attribution.record_gpu_result(
                                &self.rank1.ctx,
                                phase,
                                "nccl_remote_expert",
                                || format!("layer.{layer_idx}.nccl.remote_expert"),
                                Some(layer_idx),
                                token_index,
                                || {
                                    expert_forward_device(
                                        &self.rank1.ctx,
                                        expert,
                                        &rank1_input,
                                        token,
                                    )
                                },
                            )?,
                            &mut rank1_contrib,
                        )
                    }
                    other => {
                        bail!("routed expert {global_expert} maps to unsupported EP rank {other}")
                    }
                };
                let offset = token * self.config.hidden_size;
                attribution.record_result(
                    phase,
                    "nccl_contribution_accumulate",
                    || format!("layer.{layer_idx}.nccl.contribution_accumulate"),
                    Some(layer_idx),
                    token_index,
                    || {
                        for (dst, value) in dst[offset..offset + self.config.hidden_size]
                            .iter_mut()
                            .zip(out)
                        {
                            *dst += weight * value;
                        }
                        Ok(())
                    },
                )?;
            }
        }

        let combined = attribution.record_gpu_pair_result(
            &self.rank0.ctx,
            &self.rank1.ctx,
            phase,
            "nccl_combine",
            || format!("layer.{layer_idx}.nccl.combine"),
            Some(layer_idx),
            token_index,
            || {
                nccl.combine_f32_contributions_to_rank0(
                    &self.rank0.ctx,
                    &self.rank1.ctx,
                    &rank0_contrib,
                    &rank1_contrib,
                )
            },
        )?;
        let routed = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "nccl_combine_to_device",
            || format!("layer.{layer_idx}.nccl.combine_to_device"),
            Some(layer_idx),
            token_index,
            || {
                hidden_from_f32_host(
                    &self.rank0.ctx,
                    &combined,
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

    fn expert_forward_host(
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
}

fn expert_forward_device(
    ctx: &pegainfer_core::tensor::DeviceContext,
    expert: &ExpertMlp,
    input: &HiddenStates,
    token_idx: usize,
) -> Result<Vec<f32>> {
    activate(ctx)?;
    let token = ops::extract_vec(ctx, input, token_idx)?;
    let token_hidden = HiddenStates {
        hidden_dim: token.len,
        seq_len: 1,
        data: token.data,
    };
    let out = dense_mlp_forward(ctx, &expert.dense, &token_hidden)?;
    hidden_to_f32(ctx, &out)
}
