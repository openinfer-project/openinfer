use super::{
    backend::{EpBackendKind, parse_backend, validate_backend_and_devices},
    helpers::{append_generated_token, ensure_same_prompt_batch_rows_match},
    moe::{
        HostStagedExpertBatchPolicy, HostStagedRouteWork, NcclExpertBatchPolicy, NcclRouterPolicy,
        accumulate_host_staged_route_output, group_nccl_route_indices,
        scatter_host_staged_group_output,
    },
    routing::MoeRouteEntry,
};
use openinfer_engine::engine::FinishReason;

#[test]
fn append_generated_token_handles_eos_stop_vs_ignore() {
    // EOS hit with ignore_eos=false: stop, do not append the EOS token.
    let mut generated = vec![10, 11];
    let finish_reason = append_generated_token(&mut generated, 100_001, 100_001, false);
    assert_eq!(finish_reason, Some(FinishReason::Stop));
    assert_eq!(generated, vec![10, 11]);

    // EOS hit with ignore_eos=true: keep going, append the token.
    let mut generated = vec![10, 11];
    let finish_reason = append_generated_token(&mut generated, 100_001, 100_001, true);
    assert_eq!(finish_reason, None);
    assert_eq!(generated, vec![10, 11, 100_001]);
}

#[test]
fn same_prompt_batch_rows_must_match() {
    ensure_same_prompt_batch_rows_match(&[
        vec![11, 304, 608],
        vec![11, 304, 608],
        vec![11, 304, 608],
    ])
    .unwrap();

    let err =
        ensure_same_prompt_batch_rows_match(&[vec![11, 304, 608], vec![11, 463, 608]]).unwrap_err();
    assert!(
        err.to_string().contains("generated token index 1"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn duplicate_device_ordinals_are_rejected() {
    let err = validate_backend_and_devices(&[0, 0]).unwrap_err();

    assert!(
        err.to_string()
            .contains("two distinct CUDA device ordinals"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn ep_backend_parsing() {
    assert_eq!(parse_backend(None).unwrap(), EpBackendKind::HostStaged);
    assert_eq!(parse_backend(Some("nccl")).unwrap(), EpBackendKind::Nccl);

    let err = parse_backend(Some("pplx")).unwrap_err();
    assert!(
        err.to_string()
            .contains("supported backends: host-staged, nccl"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn host_staged_expert_batch_policy_defaults_to_batched() {
    assert_eq!(
        HostStagedExpertBatchPolicy::from_env_value(None),
        HostStagedExpertBatchPolicy::Batched
    );
    assert_eq!(
        HostStagedExpertBatchPolicy::from_env_value(Some("")),
        HostStagedExpertBatchPolicy::Batched
    );
    assert_eq!(
        HostStagedExpertBatchPolicy::from_env_value(Some("unexpected")),
        HostStagedExpertBatchPolicy::Batched
    );
}

#[test]
fn host_staged_expert_batch_policy_can_replay_serial_path() {
    for value in ["0", "false", "off", "serial", "legacy"] {
        assert_eq!(
            HostStagedExpertBatchPolicy::from_env_value(Some(value)),
            HostStagedExpertBatchPolicy::Serial
        );
    }
}

#[test]
fn nccl_expert_batch_policy_defaults_to_grouped() {
    assert_eq!(
        NcclExpertBatchPolicy::from_env_value(None),
        NcclExpertBatchPolicy::Grouped
    );
    assert_eq!(
        NcclExpertBatchPolicy::from_env_value(Some("")),
        NcclExpertBatchPolicy::Grouped
    );
    assert_eq!(
        NcclExpertBatchPolicy::from_env_value(Some("unexpected")),
        NcclExpertBatchPolicy::Grouped
    );
}

#[test]
fn nccl_expert_batch_policy_can_replay_serial_path() {
    for value in ["0", "false", "off", "serial", "legacy"] {
        assert_eq!(
            NcclExpertBatchPolicy::from_env_value(Some(value)),
            NcclExpertBatchPolicy::Serial
        );
    }
}

#[test]
fn nccl_router_policy_defaults_to_device() {
    assert_eq!(
        NcclRouterPolicy::from_env_value(None),
        NcclRouterPolicy::Device
    );
    assert_eq!(
        NcclRouterPolicy::from_env_value(Some("")),
        NcclRouterPolicy::Device
    );
    assert_eq!(
        NcclRouterPolicy::from_env_value(Some("unexpected")),
        NcclRouterPolicy::Device
    );
}

#[test]
fn nccl_router_policy_can_replay_host_path() {
    for value in ["0", "false", "off", "host", "cpu", "legacy"] {
        assert_eq!(
            NcclRouterPolicy::from_env_value(Some(value)),
            NcclRouterPolicy::Host
        );
    }
}

#[test]
fn nccl_route_groups_are_stable_by_rank_and_expert() {
    let routes = vec![
        MoeRouteEntry {
            token: 0,
            global_expert: 35,
            owner_rank: 1,
            weight: 0.6,
        },
        MoeRouteEntry {
            token: 0,
            global_expert: 3,
            owner_rank: 0,
            weight: 0.4,
        },
        MoeRouteEntry {
            token: 1,
            global_expert: 35,
            owner_rank: 1,
            weight: 0.7,
        },
    ];

    let groups: Vec<_> = group_nccl_route_indices(&routes).into_iter().collect();

    assert_eq!(groups, vec![((0, 3), vec![1]), ((1, 35), vec![0, 2])]);
}

#[test]
fn host_staged_group_scatter_matches_serial_for_distinct_rows() {
    let route_work = vec![
        HostStagedRouteWork::new(0, 7, 0, 0.5),
        HostStagedRouteWork::new(1, 7, 0, 0.25),
        HostStagedRouteWork::new(0, 9, 1, 0.2),
    ];
    let serial_outputs = vec![
        Some(vec![10.0, 11.0]),
        Some(vec![20.0, 21.0]),
        Some(vec![30.0, 31.0]),
    ];
    let mut grouped_outputs = vec![None; route_work.len()];
    scatter_host_staged_group_output(&mut grouped_outputs, &[1, 0], &[20.0, 21.0, 10.0, 11.0], 2)
        .unwrap();
    scatter_host_staged_group_output(&mut grouped_outputs, &[2], &[30.0, 31.0], 2).unwrap();

    fn combine(
        route_work: &[HostStagedRouteWork],
        mut route_outputs: Vec<Option<Vec<f32>>>,
    ) -> Vec<f32> {
        let mut rank0 = vec![0.0; 4];
        let mut rank1 = vec![0.0; 4];
        for (route_index, route) in route_work.iter().copied().enumerate() {
            let dst = if route.owner_rank == 0 {
                &mut rank0
            } else {
                &mut rank1
            };
            accumulate_host_staged_route_output(
                dst,
                route,
                route_outputs[route_index].take().unwrap(),
                2,
            )
            .unwrap();
        }
        rank0
            .into_iter()
            .zip(rank1)
            .map(|(local, remote)| local + remote)
            .collect()
    }

    let serial = combine(&route_work, serial_outputs);
    let grouped = combine(&route_work, grouped_outputs);
    assert_eq!(grouped, serial);
    for (actual, expected) in grouped.iter().zip([11.0, 11.7, 5.0, 5.25]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}
