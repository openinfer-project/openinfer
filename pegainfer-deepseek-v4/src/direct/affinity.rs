pub(super) use pegainfer_core::cpu_topology::{
    CpuId, RankThreadPlacement, RankThreadPlacementPlan, pin_current_thread_to_cpu,
};

pub(super) fn pin_scheduler_thread(placement: &RankThreadPlacementPlan) {
    let Some(cpu) = placement.scheduler_cpu() else {
        log::warn!("DeepSeek V4 scheduler CPU1 is not in the current affinity mask");
        return;
    };
    pin_current_thread_to_cpu(cpu)
        .unwrap_or_else(|err| panic!("failed to pin DeepSeek V4 scheduler to CPU {cpu}: {err:#}"));
    log::info!("pinned DeepSeek V4 scheduler to CPU {cpu}");
}

pub(super) fn pin_rank_worker_thread(placement: &RankThreadPlacement) {
    pin_current_thread_to_cpu(placement.rank_worker_cpu).unwrap_or_else(|err| {
        panic!(
            "failed to pin DeepSeek rank worker {} to CPU {}: {err:#}",
            placement.rank, placement.rank_worker_cpu
        )
    });
    log::info!(
        "pinned DeepSeek rank worker {} for CUDA device {} NUMA{} slice={} to CPU {}",
        placement.rank,
        placement.device_ordinal,
        placement.numa_node,
        placement.cpu_slice_display(),
        placement.rank_worker_cpu,
    );
}
