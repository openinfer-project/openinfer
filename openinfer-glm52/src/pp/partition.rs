//! PP8 stage → layer partition.
//!
//! At `bs=1` the decode is serial: one token walks all 78 layers across the 8
//! stages, so the partition does NOT change TPOT (the per-stage compute always
//! sums to the same total). What it governs is **per-GPU memory** — each stage
//! must hold its layers' 256 routed experts. We therefore split the 78 layers
//! into contiguous, near-equal chunks; the 3 dense layers (0..3, no routed
//! experts) fall on stage 0 naturally, which also carries the token embedding,
//! so stage 0 stays the lightest. The last stage carries the final norm +
//! lm_head. EP1 means every stage holds ALL 256 experts for its MoE layers.

use std::ops::Range;

use crate::config::GLM52_LAYERS;

/// One pipeline stage's residency: the contiguous layer range it owns plus the
/// embedding (stage 0) / final-norm+lm_head (last stage) bookends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StagePlan {
    pub(crate) stage: usize,
    pub(crate) layers: Range<usize>,
    pub(crate) owns_embed: bool,
    pub(crate) owns_head: bool,
}

impl Glm52StagePlan {
    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Partition all [`GLM52_LAYERS`] layers across `pp_world` stages as contiguous,
/// near-equal chunks. The first `GLM52_LAYERS % pp_world` stages get one extra
/// layer, so sizes differ by at most 1 (pp_world=8 → `[10,10,10,10,10,10,9,9]`).
pub(crate) fn glm52_pp_stage_plans(pp_world: usize) -> Vec<Glm52StagePlan> {
    assert!(
        pp_world > 0 && pp_world <= GLM52_LAYERS,
        "GLM5.2 PP world {pp_world} must be in 1..={GLM52_LAYERS}"
    );
    let base = GLM52_LAYERS / pp_world;
    let remainder = GLM52_LAYERS % pp_world;
    let mut plans = Vec::with_capacity(pp_world);
    let mut start = 0;
    for stage in 0..pp_world {
        let len = base + usize::from(stage < remainder);
        let end = start + len;
        plans.push(Glm52StagePlan {
            stage,
            layers: start..end,
            owns_embed: stage == 0,
            owns_head: stage == pp_world - 1,
        });
        start = end;
    }
    debug_assert_eq!(start, GLM52_LAYERS, "partition must cover every layer");
    plans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GLM52_DENSE_LAYERS;

    #[test]
    fn pp8_partition_covers_every_layer_contiguously() {
        let plans = glm52_pp_stage_plans(8);
        assert_eq!(plans.len(), 8);

        // Contiguous, gap-free cover of [0, GLM52_LAYERS).
        let mut expected_start = 0;
        for (idx, plan) in plans.iter().enumerate() {
            assert_eq!(plan.stage, idx);
            assert_eq!(plan.layers.start, expected_start, "stage {idx} start");
            expected_start = plan.layers.end;
        }
        assert_eq!(expected_start, GLM52_LAYERS);
    }

    #[test]
    fn pp8_partition_is_balanced_with_known_sizes() {
        let sizes: Vec<usize> = glm52_pp_stage_plans(8)
            .iter()
            .map(Glm52StagePlan::layer_count)
            .collect();
        // 78 = 6*10 + 2*9; the first `78 % 8 == 6` stages carry the extra layer.
        assert_eq!(sizes, vec![10, 10, 10, 10, 10, 10, 9, 9]);
        let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
        assert!(max - min <= 1, "stage sizes differ by at most one layer");
    }

    #[test]
    fn bookends_sit_on_the_end_stages_only() {
        let plans = glm52_pp_stage_plans(8);
        assert!(plans[0].owns_embed && !plans[0].owns_head);
        assert!(plans[7].owns_head && !plans[7].owns_embed);
        for plan in &plans[1..7] {
            assert!(!plan.owns_embed && !plan.owns_head);
        }
        // The 3 dense layers all land on stage 0 (it carries the embedding too).
        assert!(plans[0].layers.start == 0 && plans[0].layers.end >= GLM52_DENSE_LAYERS);
    }

    #[test]
    fn degenerate_worlds_are_well_formed() {
        let one = glm52_pp_stage_plans(1);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].layers, 0..GLM52_LAYERS);
        assert!(one[0].owns_embed && one[0].owns_head);

        let full = glm52_pp_stage_plans(GLM52_LAYERS);
        assert_eq!(full.len(), GLM52_LAYERS);
        assert!(full.iter().all(|p| p.layer_count() == 1));
    }
}
