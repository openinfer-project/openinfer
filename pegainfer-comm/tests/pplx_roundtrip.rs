//! Integration test: dispatch → recv → combine round-trip.
//!
//! Verifies that data dispatched from one rank arrives at the expected expert
//! slot on the destination rank, and that the combine path aggregates it back.
//!
//! Requires 8 GPUs with NVLink + RDMA. Run with:
//!   cargo test -p pegainfer-comm --test pplx_roundtrip -- --nocapture

use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Barrier};
use std::thread;

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use pegainfer_comm::ScalarType;
use pegainfer_comm::bootstrap::{
    EpModelShape, PplxBootstrapParams, build_intra_node_backends_for_devices,
};

const WORLD_SIZE: usize = 8;
const N_EXPERTS: usize = 384;
const TOPK: usize = 8;
const HIDDEN: usize = 7168;
const MAX_TOKENS: usize = 64;
const MAX_PRIVATE: usize = 64;
const EXPERT_PADDING: usize = 8;
const STRESS_ITERS: usize = 8;

#[test]
fn dispatch_recv_roundtrip() {
    let shape = EpModelShape {
        n_routed_experts: N_EXPERTS,
        n_activated_experts: TOPK,
        hidden_dim: HIDDEN,
    };
    let params = PplxBootstrapParams {
        max_num_tokens: MAX_TOKENS,
        expert_padding: EXPERT_PADDING,
        max_private_tokens: Some(MAX_PRIVATE),
        out_dtype: ScalarType::F32,
        nets_per_gpu: 1,
        imm_base: 0x8b00_0000,
        canonicalize_duplicate_sources: false,
    };
    let devices: Vec<usize> = (0..WORLD_SIZE).collect();
    let (backends, _resources) =
        build_intra_node_backends_for_devices(shape, &devices, params)
            .expect("bootstrap failed");

    let local_experts = N_EXPERTS / WORLD_SIZE;
    let barrier = Arc::new(Barrier::new(WORLD_SIZE));
    let errors: Vec<Option<String>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(WORLD_SIZE);
        for (rank, mut backend) in backends.into_iter().enumerate() {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || -> Option<String> {
                let ctx = CudaContext::new(rank).expect("cuda context");
                let stream = ctx.new_stream().expect("cuda stream");

                let mut x_host = vec![bf16::from_f32(0.0); MAX_TOKENS * HIDDEN];
                let mut indices_host = Vec::with_capacity(MAX_TOKENS * TOPK);
                let mut weights_host = Vec::with_capacity(MAX_TOKENS * TOPK);
                for token in 0..MAX_TOKENS {
                    let base = token * HIDDEN;
                    x_host[base] = bf16::from_f32(rank as f32);
                    x_host[base + 1] = bf16::from_f32(token as f32);
                    x_host[base + 2] =
                        bf16::from_f32((rank * MAX_TOKENS + token) as f32);
                    for dim in 3..HIDDEN {
                        x_host[base + dim] = bf16::from_f32((rank + 1) as f32 * 0.25);
                    }
                    for k in 0..TOPK {
                        let dst_rank = route_dst_rank(rank, token, k);
                        let local_expert = (token * TOPK + k) % local_experts;
                        indices_host
                            .push((dst_rank * local_experts + local_expert) as i32);
                        weights_host.push(dispatch_weight(rank, token, k));
                    }
                }
                let dummy_weights_host = vec![1.0f32; MAX_TOKENS * TOPK];

                let x = stream.clone_htod(&x_host).expect("htod x");
                let indices = stream.clone_htod(&indices_host).expect("htod indices");
                let weights = stream.clone_htod(&weights_host).expect("htod weights");
                let dummy_weights =
                    stream.clone_htod(&dummy_weights_host).expect("htod dummy weights");

                let max_recv = MAX_PRIVATE * WORLD_SIZE
                    + round_up(
                        std::cmp::max(
                            std::cmp::min(
                                MAX_TOKENS * WORLD_SIZE * TOPK
                                    + local_experts * (EXPERT_PADDING - 1),
                                MAX_TOKENS * WORLD_SIZE * local_experts,
                            ),
                            local_experts * EXPERT_PADDING,
                        ),
                        EXPERT_PADDING,
                    );

                let mut recv_tokens_per_expert =
                    stream.alloc_zeros::<i32>(local_experts).expect("alloc recv_tpe");
                let mut out_x =
                    stream.alloc_zeros::<bf16>(max_recv * HIDDEN).expect("alloc out_x");
                let mut out_weights =
                    stream.alloc_zeros::<f32>(max_recv).expect("alloc out_weights");
                stream.synchronize().expect("sync pre-dispatch");

                barrier.wait();

                // --- dispatch ---
                {
                    let cu_stream = stream.cu_stream() as u64;
                    let (x_ptr, _g0) = x.device_ptr(&stream);
                    let (idx_ptr, _g1) = indices.device_ptr(&stream);
                    let (w_ptr, _g2) = weights.device_ptr(&stream);
                    backend
                        .dispatch_send(
                            MAX_TOKENS,
                            x_ptr as *const c_void,
                            HIDDEN * 2,
                            ptr::null(),
                            0,
                            0,
                            idx_ptr as *const i32,
                            TOPK,
                            w_ptr as *const f32,
                            TOPK,
                            ptr::null(),
                            cu_stream,
                        )
                        .expect("dispatch_send");
                }
                {
                    let cu_stream = stream.cu_stream() as u64;
                    let (out_num_ptr, _g0) =
                        recv_tokens_per_expert.device_ptr_mut(&stream);
                    let (out_x_ptr, _g1) = out_x.device_ptr_mut(&stream);
                    let (out_w_ptr, _g2) = out_weights.device_ptr_mut(&stream);
                    backend
                        .dispatch_recv(
                            out_num_ptr as *mut i32,
                            out_x_ptr as *mut c_void,
                            HIDDEN * 2,
                            out_w_ptr as *mut c_void,
                            1,
                            1,
                            cu_stream,
                        )
                        .expect("dispatch_recv");
                }
                stream.synchronize().expect("sync post-dispatch");

                // Verify we received tokens
                let recv_tpe_host =
                    stream.clone_dtoh(&recv_tokens_per_expert).expect("dtoh recv_tpe");
                let total_recv: i32 = recv_tpe_host.iter().sum();
                if total_recv == 0 {
                    return Some(format!("rank {rank}: received 0 tokens"));
                }
                let out_weights_host =
                    stream.clone_dtoh(&out_weights).expect("dtoh out_weights");
                let out_x_host = stream.clone_dtoh(&out_x).expect("dtoh out_x");
                if let Err(err) = verify_dispatch_weights(
                    rank,
                    local_experts,
                    &recv_tpe_host,
                    &out_weights_host,
                    &out_x_host,
                ) {
                    return Some(err);
                }
                let expert_y_host = match build_route_tagged_expert_output(
                    &recv_tpe_host,
                    &out_weights_host,
                ) {
                    Ok(host) => host,
                    Err(err) => return Some(err),
                };
                let expert_y =
                    stream.clone_htod(&expert_y_host).expect("htod expert_y");

                // --- combine ---
                let mut out_tokens = stream
                    .alloc_zeros::<f32>(MAX_TOKENS * HIDDEN)
                    .expect("alloc out_tokens");
                {
                    let cu_stream = stream.cu_stream() as u64;
                    let (expert_ptr, _g) = expert_y.device_ptr(&stream);
                    backend
                        .combine_send(
                            expert_ptr as *const c_void,
                            HIDDEN * 2,
                            cu_stream,
                        )
                        .expect("combine_send");
                }
                {
                    let cu_stream = stream.cu_stream() as u64;
                    let (out_ptr, _g0) = out_tokens.device_ptr_mut(&stream);
                    let (idx_ptr, _g1) = indices.device_ptr(&stream);
                    let (w_ptr, _g2) = dummy_weights.device_ptr(&stream);
                    backend
                        .combine_recv(
                            MAX_TOKENS,
                            0,
                            ScalarType::BF16,
                            out_ptr as *mut c_void,
                            HIDDEN,
                            idx_ptr as *const i32,
                            TOPK,
                            w_ptr as *const f32,
                            TOPK,
                            ptr::null(),
                            false,
                            cu_stream,
                        )
                        .expect("combine_recv");
                }
                stream.synchronize().expect("sync post-combine");
                if let Err(err) = verify_route_tagged_combine(
                    rank,
                    &stream.clone_dtoh(&out_tokens).expect("dtoh out_tokens"),
                ) {
                    return Some(err);
                }

                barrier.wait();

                for _ in 0..STRESS_ITERS {
                    let cu_stream = stream.cu_stream() as u64;
                    let (x_ptr, _g0) = x.device_ptr(&stream);
                    let (idx_ptr, _g1) = indices.device_ptr(&stream);
                    let (w_ptr, _g2) = weights.device_ptr(&stream);
                    backend
                        .dispatch_send(
                            MAX_TOKENS,
                            x_ptr as *const c_void,
                            HIDDEN * 2,
                            ptr::null(),
                            0,
                            0,
                            idx_ptr as *const i32,
                            TOPK,
                            w_ptr as *const f32,
                            TOPK,
                            ptr::null(),
                            cu_stream,
                        )
                        .expect("stress dispatch_send");

                    {
                        let (out_num_ptr, _g3) =
                            recv_tokens_per_expert.device_ptr_mut(&stream);
                        let (out_x_ptr, _g4) = out_x.device_ptr_mut(&stream);
                        let (out_w_ptr, _g5) = out_weights.device_ptr_mut(&stream);
                        backend
                            .dispatch_recv(
                                out_num_ptr as *mut i32,
                                out_x_ptr as *mut c_void,
                                HIDDEN * 2,
                                out_w_ptr as *mut c_void,
                                1,
                                1,
                                cu_stream,
                            )
                            .expect("stress dispatch_recv");
                    }

                    let (expert_ptr, _g6) = out_x.device_ptr(&stream);
                    backend
                        .combine_send(
                            expert_ptr as *const c_void,
                            HIDDEN * 2,
                            cu_stream,
                        )
                        .expect("stress combine_send");

                    {
                        let (out_ptr, _g7) = out_tokens.device_ptr_mut(&stream);
                        let (idx_ptr, _g8) = indices.device_ptr(&stream);
                        let (w_ptr, _g9) = dummy_weights.device_ptr(&stream);
                        backend
                            .combine_recv(
                                MAX_TOKENS,
                                0,
                                ScalarType::BF16,
                                out_ptr as *mut c_void,
                                HIDDEN,
                                idx_ptr as *const i32,
                                TOPK,
                                w_ptr as *const f32,
                                TOPK,
                                ptr::null(),
                                false,
                                cu_stream,
                            )
                            .expect("stress combine_recv");
                    }
                }
                stream.synchronize().expect("sync post-stress");
                if let Err(err) = verify_passthrough_combine(
                    rank,
                    &stream.clone_dtoh(&out_tokens).expect("dtoh stress out_tokens"),
                ) {
                    return Some(err);
                }

                eprintln!(
                    "rank {rank}: dispatch+combine ok, received {total_recv} tokens"
                );
                None
            }));
        }
        handles.into_iter().map(|h| h.join().expect("rank thread panicked")).collect()
    });

    let failures: Vec<_> = errors.into_iter().flatten().collect();
    assert!(failures.is_empty(), "failures: {failures:?}");
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn dispatch_weight(rank: usize, token: usize, route: usize) -> f32 {
    (rank * 10_000 + token * 100 + route + 1) as f32
}

fn route_dst_rank(src_rank: usize, token: usize, route: usize) -> usize {
    let anchor = (src_rank + token) % WORLD_SIZE;
    if route < TOPK / 2 { anchor } else { (anchor + route - TOPK / 2 + 1) % WORLD_SIZE }
}

fn verify_dispatch_weights(
    rank: usize,
    local_experts: usize,
    recv_tpe: &[i32],
    out_weights: &[f32],
    out_x: &[bf16],
) -> Result<(), String> {
    let mut expected = vec![Vec::new(); local_experts];
    for src_rank in 0..WORLD_SIZE {
        for token in 0..MAX_TOKENS {
            for route in 0..TOPK {
                let dst_rank = route_dst_rank(src_rank, token, route);
                if dst_rank != rank {
                    continue;
                }
                let local_expert = (token * TOPK + route) % local_experts;
                expected[local_expert].push(dispatch_weight(src_rank, token, route));
            }
        }
    }

    let mut padded_offset = 0usize;
    for expert in 0..local_experts {
        let count = recv_tpe[expert] as usize;
        let mut got = out_weights[padded_offset..padded_offset + count].to_vec();
        for (row, weight) in got.iter().copied().enumerate() {
            let (src_rank, token, route) = decode_dispatch_weight(weight)?;
            let row_idx = padded_offset + row;
            let got_rank = out_x[row_idx * HIDDEN].to_f32() as usize;
            let got_token = out_x[row_idx * HIDDEN + 1].to_f32() as usize;
            if got_rank != src_rank || got_token != token {
                return Err(format!(
                    "rank {rank} expert {expert} row {row}: hidden/weight mismatch: hidden=({got_rank},{got_token}) weight=({src_rank},{token},{route})"
                ));
            }
        }
        got.sort_by(f32::total_cmp);
        expected[expert].sort_by(f32::total_cmp);
        if got != expected[expert] {
            return Err(format!(
                "rank {rank} expert {expert}: dispatch weights mismatch: got {:?}, expected {:?}",
                &got[..got.len().min(8)],
                &expected[expert][..expected[expert].len().min(8)]
            ));
        }
        padded_offset += round_up(count, EXPERT_PADDING);
    }
    Ok(())
}

fn decode_dispatch_weight(weight: f32) -> Result<(usize, usize, usize), String> {
    let encoded = weight.round() as usize;
    if encoded == 0 {
        return Err("dispatch weight must be positive".to_string());
    }
    let src_rank = encoded / 10_000;
    let rest = encoded % 10_000;
    let token_route = rest - 1;
    let token = token_route / 100;
    let route = token_route % 100;
    if src_rank >= WORLD_SIZE || token >= MAX_TOKENS || route >= TOPK {
        return Err(format!(
            "decoded dispatch weight {weight} out of range: ({src_rank},{token},{route})"
        ));
    }
    Ok((src_rank, token, route))
}

fn build_route_tagged_expert_output(
    recv_tpe: &[i32],
    out_weights: &[f32],
) -> Result<Vec<bf16>, String> {
    let mut expert_y = vec![bf16::from_f32(0.0); out_weights.len() * HIDDEN];
    let mut padded_offset = 0usize;
    for count in recv_tpe.iter().map(|count| *count as usize) {
        for row in 0..count {
            let row_idx = padded_offset + row;
            let (src_rank, token, route) =
                decode_dispatch_weight(out_weights[row_idx])?;
            let base = row_idx * HIDDEN;
            expert_y[base] = bf16::from_f32(src_rank as f32);
            expert_y[base + 1] = bf16::from_f32(token as f32);
            expert_y[base + 2] = bf16::from_f32(route as f32);
            expert_y[base + 3] = bf16::from_f32(1.0);
        }
        padded_offset += round_up(count, EXPERT_PADDING);
    }
    Ok(expert_y)
}

fn verify_route_tagged_combine(rank: usize, out_tokens: &[f32]) -> Result<(), String> {
    let route_sum = (0..TOPK).sum::<usize>() as f32;
    for token in 0..MAX_TOKENS {
        let expected = [
            rank as f32 * TOPK as f32,
            token as f32 * TOPK as f32,
            route_sum,
            TOPK as f32,
        ];
        for (dim, expected) in expected.into_iter().enumerate() {
            let got = out_tokens[token * HIDDEN + dim];
            if (got - expected).abs() > 0.0 {
                return Err(format!(
                    "rank {rank} token {token} dim {dim}: combine mismatch got {got}, expected {expected}"
                ));
            }
        }
    }
    Ok(())
}

fn verify_passthrough_combine(rank: usize, out_tokens: &[f32]) -> Result<(), String> {
    for token in 0..MAX_TOKENS {
        let expected = [
            rank as f32 * TOPK as f32,
            token as f32 * TOPK as f32,
            bf16::from_f32((rank * MAX_TOKENS + token) as f32).to_f32() * TOPK as f32,
            (rank + 1) as f32 * 0.25 * TOPK as f32,
        ];
        for (dim, expected) in expected.into_iter().enumerate() {
            let got = out_tokens[token * HIDDEN + dim];
            if (got - expected).abs() > 0.0 {
                return Err(format!(
                    "rank {rank} token {token} dim {dim}: stress combine mismatch got {got}, expected {expected}"
                ));
            }
        }
    }
    Ok(())
}
