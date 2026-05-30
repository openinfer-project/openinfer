use std::path::PathBuf;

use pegainfer_core::{engine::EpBackend, parallel::ParallelConfig};

use crate::runner::affinity::KimiRankThreadPlacementPlan;
use crate::runner::worker::KimiK2RankPlacement;
use crate::weights::{KimiRankSlicedLoadPlan, KimiRankWeightNames};

#[derive(Clone, Debug)]
pub(crate) struct KimiK2RunnerConfig {
    pub model_path: PathBuf,
    pub parallel: ParallelConfig,
    pub local_dims: crate::config::KimiLocalDims,
    pub rank_weight_names: Vec<KimiRankWeightNames>,
    pub rank_sliced_load_plans: Vec<KimiRankSlicedLoadPlan>,
    pub placements: Vec<KimiK2RankPlacement>,
    pub(crate) thread_placement: KimiRankThreadPlacementPlan,
    pub(crate) pplx_thread_placement: pegainfer_core::cpu_topology::RankThreadPlacementPlan,
    pub enable_cuda_graph: bool,
    pub ep_backend: EpBackend,
}
