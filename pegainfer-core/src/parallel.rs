//! Model-agnostic parallel topology types.

/// Pure parallel topology. No model-specific fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelConfig {
    pub tp_world: usize,
    pub dp_world: usize,
    pub ep_world: usize,
}

impl ParallelConfig {
    #[must_use]
    pub fn new(tp_world: usize, dp_world: usize) -> Self {
        assert!(tp_world > 0 && dp_world > 0);
        Self {
            tp_world,
            dp_world,
            ep_world: tp_world * dp_world,
        }
    }

    #[must_use]
    pub fn coord(&self, global_rank: usize) -> RankCoord {
        assert!(global_rank < self.ep_world);
        RankCoord {
            global_rank,
            tp_rank: global_rank % self.tp_world,
            dp_rank: global_rank / self.tp_world,
            ep_rank: global_rank,
        }
    }

    #[must_use]
    pub fn tp_group(&self, dp_rank: usize) -> std::ops::Range<usize> {
        let start = dp_rank * self.tp_world;
        start..start + self.tp_world
    }
}

/// A rank's coordinate in the TP×DP×EP grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankCoord {
    pub global_rank: usize,
    pub tp_rank: usize,
    pub dp_rank: usize,
    pub ep_rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_maps_rank_to_tp_dp_ep() {
        // (tp_size, dp_size, rank) -> (tp_rank, dp_rank, ep_rank)
        let cases = [(8, 1, 3, 3, 0, 3), (1, 8, 3, 0, 3, 3), (2, 4, 5, 1, 2, 5)];
        for (tp, dp, rank, tp_rank, dp_rank, ep_rank) in cases {
            let cfg = ParallelConfig::new(tp, dp);
            assert_eq!(cfg.ep_world, tp * dp, "tp{tp} dp{dp}");
            let c = cfg.coord(rank);
            assert_eq!(
                (c.tp_rank, c.dp_rank, c.ep_rank),
                (tp_rank, dp_rank, ep_rank),
                "tp{tp} dp{dp} rank{rank}"
            );
        }
        // tp_group spans the contiguous tp ranks inside a dp group
        assert_eq!(ParallelConfig::new(2, 4).tp_group(2), 4..6);
    }
}
