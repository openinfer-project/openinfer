use super::{EpBackendKind, decode_graph_blockers};

#[test]
fn nccl_readiness_reports_only_remaining_graph_blockers() {
    let blockers = decode_graph_blockers(EpBackendKind::Nccl);
    let ids: Vec<_> = blockers.iter().map(|blocker| blocker.id).collect();

    assert_eq!(
        ids,
        vec![
            "nccl_route_plan_built_on_host",
            "nccl_route_plan_replay_host_directed",
        ]
    );
}
