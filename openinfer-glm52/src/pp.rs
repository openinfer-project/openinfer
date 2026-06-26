//! GLM5.2 PP8 decode runtime spine.
//!
//! Eight pipeline stages, one GPU each, serialized by device-memory flags over
//! NVLink P2P rather than NCCL or stream/event edges. Slice 0 builds and measures
//! the stage-boundary `L_send` handoff in isolation (dummy compute); later slices
//! replace `dummy_burn` with real per-stage layers. See
//! `docs/models/glm52/pp-decode.md`.

mod partition;
mod peer;
mod runtime;
mod stage_graph;

pub(crate) use partition::{Glm52StagePlan, glm52_pp_stage_plans};
pub use runtime::{Glm52PpHopStats, Glm52PpSpineConfig, Glm52PpSpineReport, run_pp_p2p_spine};
