use std::path::PathBuf;

use pegainfer_core::parallel::ParallelConfig;

use crate::runner::affinity::KimiRankThreadPlacementPlan;
use crate::runner::worker::KimiK2RankPlacement;
use crate::weights::{KimiK2WeightManifest, KimiRankSlicedLoadPlan, KimiRankWeightNames};

#[derive(Clone, Debug)]
pub struct KimiK2RunnerConfig {
    pub model_path: PathBuf,
    pub parallel: ParallelConfig,
    pub local_dims: crate::config::KimiLocalDims,
    pub weight_manifest: KimiK2WeightManifest,
    pub rank_weight_names: Vec<KimiRankWeightNames>,
    pub rank_sliced_load_plans: Vec<KimiRankSlicedLoadPlan>,
    pub placements: Vec<KimiK2RankPlacement>,
    pub(crate) thread_placement: KimiRankThreadPlacementPlan,
    #[cfg(feature = "pplx-ep")]
    pub(crate) pplx_thread_placement: pegainfer_core::cpu_topology::RankThreadPlacementPlan,
    pub enable_cuda_graph: bool,
}
