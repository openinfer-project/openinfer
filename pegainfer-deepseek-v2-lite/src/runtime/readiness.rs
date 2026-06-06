use anyhow::{Result, bail, ensure};

use super::{
    DeepSeekV2LiteEp2Generator,
    backend::{EpBackendKind, EpBackendRuntime},
    types::{
        DecodeGraphBlocker, DecodeGraphReadinessMetrics, DecodeGraphReadinessReport,
        GenerationStats,
    },
};

impl DeepSeekV2LiteEp2Generator {
    pub fn decode_graph_readiness_report(
        &self,
        stats: &GenerationStats,
        batch_size: usize,
        run_nccl_graph_smoke: bool,
    ) -> Result<DecodeGraphReadinessReport> {
        let backend = self.backend.kind();
        ensure!(
            stats.ep_backend == backend.as_str(),
            "DeepSeek-V2-Lite graph readiness stats backend mismatch: stats={}, runtime={}",
            stats.ep_backend,
            backend.as_str()
        );
        let nccl_graph_smoke = if run_nccl_graph_smoke {
            match &self.backend {
                EpBackendRuntime::Nccl(nccl) => {
                    let report = nccl.graph_smoke_all_reduce_f32(&self.rank0.ctx, &self.rank1.ctx);
                    ensure!(
                        report.verified(),
                        "DeepSeek-V2-Lite --nccl-graph-smoke failed: {}",
                        report.failure_summary()
                    );
                    Some(report)
                }
                EpBackendRuntime::HostStaged => bail!(
                    "DeepSeek-V2-Lite --nccl-graph-smoke requires PEGAINFER_DSV2_LITE_EP_BACKEND=nccl"
                ),
            }
        } else {
            None
        };
        Ok(DecodeGraphReadinessReport {
            schema: 1,
            backend: stats.ep_backend.clone(),
            batch_size,
            full_decode_capture_ready: false,
            status: decode_graph_readiness_status(backend),
            blockers: decode_graph_blockers(backend),
            metrics: DecodeGraphReadinessMetrics {
                host_dispatch_calls: stats.host_dispatch_calls,
                host_combine_calls: stats.host_combine_calls,
                host_dispatch_elements: stats.host_dispatch_elements,
                host_combine_elements: stats.host_combine_elements,
                nccl_dense_exchange_calls: stats.nccl_dense_exchange_calls,
                nccl_combine_calls: stats.nccl_combine_calls,
                nccl_dense_exchange_elements: stats.nccl_dense_exchange_elements,
                nccl_combine_elements: stats.nccl_combine_elements,
                nccl_dispatch_local_routes: stats.nccl_dispatch_local_routes,
                nccl_dispatch_remote_routes: stats.nccl_dispatch_remote_routes,
                nccl_combine_routes: stats.nccl_combine_routes,
            },
            nccl_graph_smoke_requested: run_nccl_graph_smoke,
            nccl_graph_smoke,
            claim_boundary: "This is a graph-readiness diagnostic for the covered DeepSeek-V2-Lite EP2 decode attribution gate. A successful NCCL f32 smoke proves only basic preallocated collective capture/replay on this runtime; it is not full decode CUDA Graph coverage or a performance claim.",
        })
    }
}

fn decode_graph_readiness_status(backend: EpBackendKind) -> &'static str {
    match backend {
        EpBackendKind::HostStaged => "not_applicable_host_staged_backend",
        EpBackendKind::Nccl => "blocked_full_decode_path",
    }
}

fn decode_graph_blockers(backend: EpBackendKind) -> Vec<DecodeGraphBlocker> {
    match backend {
        EpBackendKind::HostStaged => vec![
            DecodeGraphBlocker {
                id: "host_staged_route_and_dispatch_on_host",
                source: "runtime/moe.rs::moe_forward_host_staged",
                reason: "routing, per-route expert dispatch, and contribution accumulation are intentionally host-staged",
            },
            DecodeGraphBlocker {
                id: "host_staged_hidden_d2h_and_h2d",
                source: "host_ops.rs::hidden_to_bf16 / hidden_from_f32_host",
                reason: "the baseline path copies hidden states through host memory and synchronizes around those copies",
            },
        ],
        EpBackendKind::Nccl => vec![
            DecodeGraphBlocker {
                id: "nccl_dense_exchange_allocates_per_call",
                source: "nccl_backend.rs::dense_all_reduce_rank0_hidden_to_rank1",
                reason: "rank0/rank1 receive and zero-send buffers are allocated inside each dense exchange",
            },
            DecodeGraphBlocker {
                id: "nccl_dense_exchange_syncs_rank_streams",
                source: "nccl_backend.rs::dense_all_reduce_rank0_hidden_to_rank1",
                reason: "both rank streams are synchronized after every dense exchange",
            },
            DecodeGraphBlocker {
                id: "nccl_route_iteration_on_host",
                source: "runtime/moe.rs::moe_forward_nccl",
                reason: "expert routing and per-route loop decisions stay on the host",
            },
            DecodeGraphBlocker {
                id: "nccl_contribution_accumulation_on_host",
                source: "runtime/moe.rs::moe_forward_nccl",
                reason: "local and remote expert outputs are copied back and accumulated in host vectors",
            },
            DecodeGraphBlocker {
                id: "nccl_combine_h2d_contribution_copy",
                source: "nccl_backend.rs::combine_f32_contributions_to_rank0",
                reason: "rank contributions are copied from host to device for each combine call",
            },
            DecodeGraphBlocker {
                id: "nccl_combine_allocates_per_call",
                source: "nccl_backend.rs::combine_f32_contributions_to_rank0",
                reason: "send and receive buffers are allocated inside each combine call",
            },
            DecodeGraphBlocker {
                id: "nccl_combine_syncs_rank_streams",
                source: "nccl_backend.rs::combine_f32_contributions_to_rank0",
                reason: "both rank streams are synchronized after every combine collective",
            },
            DecodeGraphBlocker {
                id: "nccl_combine_d2h_result_copy",
                source: "nccl_backend.rs::combine_f32_contributions_to_rank0",
                reason: "the combined rank0 result is copied back to host before being uploaded again",
            },
        ],
    }
}
