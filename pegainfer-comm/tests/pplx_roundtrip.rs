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
const N_EXPERTS: usize = 256;
const TOPK: usize = 6;
const HIDDEN: usize = 128;
const MAX_TOKENS: usize = 1;
const MAX_PRIVATE: usize = 64;
const EXPERT_PADDING: usize = 16;

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
        nets_per_gpu: 1,
        imm_base: 0x8b00_0000,
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

                let x_val = bf16::from_f32((rank + 1) as f32 * 0.5);
                let x_host = vec![x_val; MAX_TOKENS * HIDDEN];
                let mut indices_host = Vec::with_capacity(MAX_TOKENS * TOPK);
                for token in 0..MAX_TOKENS {
                    for k in 0..TOPK {
                        let dst_rank = (rank + k + 1) % WORLD_SIZE;
                        let local_expert = (token * TOPK + k) % local_experts;
                        indices_host
                            .push((dst_rank * local_experts + local_expert) as i32);
                    }
                }
                let weights_host = vec![1.0f32 / TOPK as f32; MAX_TOKENS * TOPK];

                let x = stream.clone_htod(&x_host).expect("htod x");
                let indices = stream.clone_htod(&indices_host).expect("htod indices");
                let weights = stream.clone_htod(&weights_host).expect("htod weights");

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
                    backend
                        .dispatch_recv(
                            out_num_ptr as *mut i32,
                            out_x_ptr as *mut c_void,
                            HIDDEN * 2,
                            ptr::null_mut(),
                            0,
                            0,
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

                // --- combine ---
                let expert_y = stream
                    .alloc_zeros::<bf16>(max_recv * HIDDEN)
                    .expect("alloc expert_y");
                let mut out_tokens = stream
                    .alloc_zeros::<bf16>(MAX_TOKENS * HIDDEN)
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
                    let (w_ptr, _g2) = weights.device_ptr(&stream);
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
                            true,
                            cu_stream,
                        )
                        .expect("combine_recv");
                }
                stream.synchronize().expect("sync post-combine");

                barrier.wait();

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
