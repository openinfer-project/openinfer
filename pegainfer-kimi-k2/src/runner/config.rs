use std::path::PathBuf;

use crate::runner::affinity::KimiRankThreadPlacementPlan;
use crate::runner::worker::KimiK2RankPlacement;
use crate::weights::{
    KimiK2WeightManifest, KimiRankShardPlan, KimiRankSlicedLoadPlan, KimiRankWeightNames,
    KimiRankWeightPlan,
};

#[derive(Clone, Debug)]
pub struct KimiK2RunnerConfig {
    pub model_path: PathBuf,
    pub weight_manifest: KimiK2WeightManifest,
    pub rank_weight_plans: Vec<KimiRankWeightPlan>,
    pub rank_weight_names: Vec<KimiRankWeightNames>,
    pub rank_shard_plans: Vec<KimiRankShardPlan>,
    pub rank_sliced_load_plans: Vec<KimiRankSlicedLoadPlan>,
    pub placements: Vec<KimiK2RankPlacement>,
    pub(crate) thread_placement: KimiRankThreadPlacementPlan,
    pub enable_cuda_graph: bool,
}
