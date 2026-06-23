use half::bf16;
use openinfer_core::tensor::{DeviceContext, DeviceMatrix, HiddenStates};
use openinfer_kernels::ops::{
    Dsv2LiteRouterOutput, dsv2_lite_accumulate_fixed_expert_into,
    dsv2_lite_router_softmax_topk_into,
};

use crate::{
    config::test_lite_config,
    host_ops::{gate_logits_host, topk_softmax_routes},
};

#[test]
fn device_router_matches_host_softmax_topk_rule() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let mut config = test_lite_config();
    config.hidden_size = 4;
    config.n_routed_experts = 8;
    config.num_experts_per_token = 6;

    let hidden_host = bf16_vec(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5]);
    let gate_host = router_gate_fixture();
    let hidden = HiddenStates {
        data: ctx.stream.clone_htod(&hidden_host).expect("hidden H2D"),
        hidden_dim: config.hidden_size,
        seq_len: 2,
    };
    let gate = DeviceMatrix::from_host(
        &ctx,
        &gate_host,
        config.n_routed_experts,
        config.hidden_size,
    )
    .expect("gate H2D");
    let mut topk_weight = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk weight");
    let mut topk_idx = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk idx");

    dsv2_lite_router_softmax_topk_into(
        &ctx,
        &hidden,
        &gate,
        config.num_experts_per_token,
        &mut Dsv2LiteRouterOutput {
            topk_weight: &mut topk_weight,
            topk_idx: &mut topk_idx,
        },
    )
    .expect("device router");
    let got_idx = ctx.stream.clone_dtoh(&topk_idx).expect("idx D2H");
    let got_weight = ctx.stream.clone_dtoh(&topk_weight).expect("weight D2H");
    ctx.sync().expect("sync router outputs");

    let gate_host_f32: Vec<_> = gate_host.iter().map(|value| value.to_f32()).collect();
    let logits = gate_logits_host(&config, &hidden_host, &gate_host_f32);
    let expected = topk_softmax_routes(&config, &logits, hidden.seq_len);

    for token in 0..hidden.seq_len {
        for route in 0..config.num_experts_per_token {
            let offset = token * config.num_experts_per_token + route;
            let (expected_idx, expected_weight) = expected[token][route];
            assert_eq!(got_idx[offset], expected_idx as i32);
            assert_close(got_weight[offset], expected_weight, 1.0e-6);
        }
    }
}

#[test]
fn device_router_handles_single_zero_decode_row() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let mut config = test_lite_config();
    config.hidden_size = 4;
    config.n_routed_experts = 8;
    config.num_experts_per_token = 6;

    let hidden_host = bf16_vec(&[0.0, 0.0, 0.0, 0.0]);
    let gate_host = router_gate_fixture();
    let hidden = HiddenStates {
        data: ctx.stream.clone_htod(&hidden_host).expect("hidden H2D"),
        hidden_dim: config.hidden_size,
        seq_len: 1,
    };
    let gate = DeviceMatrix::from_host(
        &ctx,
        &gate_host,
        config.n_routed_experts,
        config.hidden_size,
    )
    .expect("gate H2D");
    let mut topk_weight = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk weight");
    let mut topk_idx = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk idx");

    dsv2_lite_router_softmax_topk_into(
        &ctx,
        &hidden,
        &gate,
        config.num_experts_per_token,
        &mut Dsv2LiteRouterOutput {
            topk_weight: &mut topk_weight,
            topk_idx: &mut topk_idx,
        },
    )
    .expect("device router");
    let got_idx = ctx.stream.clone_dtoh(&topk_idx).expect("idx D2H");
    let got_weight = ctx.stream.clone_dtoh(&topk_weight).expect("weight D2H");
    ctx.sync().expect("sync router outputs");

    let gate_host_f32: Vec<_> = gate_host.iter().map(|value| value.to_f32()).collect();
    let logits = gate_logits_host(&config, &hidden_host, &gate_host_f32);
    let expected = topk_softmax_routes(&config, &logits, hidden.seq_len);
    for route in 0..config.num_experts_per_token {
        let (expected_idx, expected_weight) = expected[0][route];
        assert_eq!(got_idx[route], expected_idx as i32);
        assert_close(got_weight[route], expected_weight, 1.0e-6);
    }
}

#[test]
fn fixed_expert_accumulate_masks_inactive_experts_and_accumulates_matches() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden_dim = 3;
    let seq_len = 2;
    let topk = 3;
    let expert_output = expert_output_fixture(&ctx, hidden_dim, seq_len);
    let topk_weight = ctx
        .stream
        .clone_htod(&[0.25f32, 0.5, 0.25, 0.1, 0.2, 0.7])
        .expect("route weights H2D");
    let topk_idx = ctx
        .stream
        .clone_htod(&[2i32, 5, 7, 1, 5, 6])
        .expect("route idx H2D");
    let mut accum = ctx
        .stream
        .alloc_zeros::<f32>(hidden_dim * seq_len)
        .expect("accum");

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        3,
        topk,
        &mut accum,
    )
    .expect("inactive expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("inactive D2H");
    ctx.sync().expect("sync inactive accumulate");
    assert_eq!(got, vec![0.0; hidden_dim * seq_len]);

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        5,
        topk,
        &mut accum,
    )
    .expect("active expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("active D2H");
    ctx.sync().expect("sync active accumulate");
    assert_vec_close(&got, &[0.5, 1.0, 1.5, 0.8, 1.0, 1.2], 1.0e-6);

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        2,
        topk,
        &mut accum,
    )
    .expect("second active expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("second active D2H");
    ctx.sync().expect("sync second active accumulate");
    assert_vec_close(&got, &[0.75, 1.5, 2.25, 0.8, 1.0, 1.2], 1.0e-6);
}

#[test]
fn fixed_expert_accumulate_ignores_zero_weight_routes() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden_dim = 3;
    let seq_len = 2;
    let topk = 3;
    let expert_output = expert_output_fixture(&ctx, hidden_dim, seq_len);
    let topk_weight = ctx
        .stream
        .clone_htod(&[1.0f32, 0.0, 0.0, 0.0, 0.5, 0.5])
        .expect("route weights H2D");
    let topk_idx = ctx
        .stream
        .clone_htod(&[4i32, 1, 2, 4, 3, 2])
        .expect("route idx H2D");
    let mut accum = ctx
        .stream
        .alloc_zeros::<f32>(hidden_dim * seq_len)
        .expect("accum");

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        4,
        topk,
        &mut accum,
    )
    .expect("all-token fixed expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("accum D2H");
    ctx.sync().expect("sync accumulate");
    assert_vec_close(&got, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], 1.0e-6);
}

fn router_gate_fixture() -> Vec<bf16> {
    bf16_vec(&[
        0.25, 0.5, -0.25, 1.0, -0.5, 0.75, 0.25, -0.25, 1.0, -1.0, 0.5, 0.25, -0.75, 0.5, 1.0,
        -0.5, 0.5, 0.0, -1.0, 0.75, -1.25, 0.25, 0.75, 0.5, 0.0, -0.5, 1.25, -0.75, 0.75, 0.25,
        -0.25, 0.5,
    ])
}

fn expert_output_fixture(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> HiddenStates {
    HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .expect("expert output H2D"),
        hidden_dim,
        seq_len,
    }
}

fn bf16_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn assert_vec_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "value mismatch at {idx}: got {got}, expected {expected}"
        );
    }
}

fn assert_close(got: f32, expected: f32, tolerance: f32) {
    assert!(
        (got - expected).abs() <= tolerance,
        "got {got}, expected {expected}, tolerance {tolerance}"
    );
}
