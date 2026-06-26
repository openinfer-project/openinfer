//! GLM5.2 DeepEP shape contract.
//!
//! The current `openinfer-kernels` DeepEP C shim is baked for Kimi-K2. GLM5.2
//! must keep its own dimensions explicit until a GLM-shaped shim exists, or a
//! future MoE path could accidentally pass 6144-wide buffers into a 7168-wide
//! communicator.

use anyhow::{Result, ensure};

use crate::config::{GLM52_HIDDEN, GLM52_ROUTED_EXPERTS, GLM52_TOPK};

pub(crate) const GLM52_EP_WORLD: usize = 8;
pub(crate) const GLM52_LOCAL_EXPERTS: usize = GLM52_ROUTED_EXPERTS / GLM52_EP_WORLD;
pub(crate) const GLM52_DEEPEP_EXPERT_ALIGNMENT: usize = 64;
pub(crate) const GLM52_DEEPEP_DEVICE_SMS: usize = 132;
pub(crate) const GLM52_DEEPEP_DECODE_BATCH_CAP: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DeepEpShape {
    pub(crate) ep_world: usize,
    pub(crate) routed_experts: usize,
    pub(crate) local_experts: usize,
    pub(crate) topk: usize,
    pub(crate) hidden: usize,
    pub(crate) expert_alignment: usize,
    pub(crate) device_sms: usize,
    pub(crate) decode_max_tokens_per_rank: usize,
}

impl Glm52DeepEpShape {
    pub(crate) fn tp1_dp8_h200() -> Self {
        Self {
            ep_world: GLM52_EP_WORLD,
            routed_experts: GLM52_ROUTED_EXPERTS,
            local_experts: GLM52_LOCAL_EXPERTS,
            topk: GLM52_TOPK,
            hidden: GLM52_HIDDEN,
            expert_alignment: GLM52_DEEPEP_EXPERT_ALIGNMENT,
            device_sms: GLM52_DEEPEP_DEVICE_SMS,
            decode_max_tokens_per_rank: GLM52_DEEPEP_DECODE_BATCH_CAP,
        }
    }

    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            self.ep_world == GLM52_EP_WORLD
                && self.routed_experts == GLM52_ROUTED_EXPERTS
                && self.local_experts == GLM52_LOCAL_EXPERTS
                && self.topk == GLM52_TOPK
                && self.hidden == GLM52_HIDDEN,
            "GLM5.2 DeepEP shape drifted from model constants: {self:?}"
        );
        ensure!(
            self.routed_experts.is_multiple_of(self.ep_world)
                && self.local_experts == self.routed_experts / self.ep_world,
            "GLM5.2 DeepEP EP{}/experts{} cannot produce local_experts={}",
            self.ep_world,
            self.routed_experts,
            self.local_experts
        );
        ensure!(
            self.expert_alignment > 0 && self.decode_max_tokens_per_rank > 0,
            "GLM5.2 DeepEP shape has invalid capacities: {self:?}"
        );
        Ok(())
    }

    pub(crate) fn decode_capacity(self) -> Result<Glm52DeepEpDecodeCapacity> {
        self.validate()?;
        let worst_recv_tokens = self.ep_world * self.decode_max_tokens_per_rank;
        Ok(Glm52DeepEpDecodeCapacity {
            shape: self,
            worst_recv_tokens,
            worst_expanded_tokens: self.expanded_rows_for_recv_tokens(worst_recv_tokens),
            src_metadata_len: worst_recv_tokens * (self.topk + 2),
            rank_count_len: self.device_sms * self.ep_world,
        })
    }

    fn expanded_rows_for_recv_tokens(self, recv_tokens: usize) -> usize {
        align_up(
            recv_tokens * self.topk.min(self.local_experts)
                + (self.expert_alignment - 1) * self.local_experts,
            self.expert_alignment,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52DeepEpDecodeCapacity {
    pub(crate) shape: Glm52DeepEpShape,
    pub(crate) worst_recv_tokens: usize,
    pub(crate) worst_expanded_tokens: usize,
    pub(crate) src_metadata_len: usize,
    pub(crate) rank_count_len: usize,
}

const fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}
