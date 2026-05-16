//! Single-process intra-node bootstrap for the pplx-garden EP backend.
//!
//! Builds one `Vec<EpBackend>` (length = world_size) ready to hand to
//! [`super::scheduler::DeepSeekV4DirectGenerator::enable_pplx`]. Because all
//! 8 ranks of DSv4 direct live in one process, this skips the per-process
//! pickle/FD rendezvous that the upstream Python bootstrap (under
//! `pegainfer-comm/python/pplx_garden/kernels/p2p_all_to_all.py`) needs and
//! instead shares CUMem allocation handles by `Arc` clone, mapping each
//! peer's send/recv/sync buffer directly into every other rank's device VA.
//!
//! # Threading model
//!
//! Each rank gets its own worker thread for the whole bootstrap. Each
//! thread calls `cudaSetDevice` once at entry, so every CUDA call inside
//! (CUMem alloc, host alloc, MR register, peer map, EpBackend::new) runs
//! against the correct device context. This replaces the previous
//! main-thread loop that had to sprinkle `cudaSetDevice` everywhere and
//! still missed cases — see the dispatch_recv ILLEGAL_ADDRESS bug.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result, bail};
use pegainfer_comm::raw::cuda_lib::cumem::{
    CUAllocHandle, CUMemAllocHandle, CUMemHandleKind, CUMemMapping,
};
use pegainfer_comm::raw::cuda_lib::{CudaDeviceId, CudaHostMemory, Device};
use pegainfer_comm::raw::fabric_lib::api::{
    DomainAddress, MemoryRegionDescriptor, MemoryRegionHandle,
};
use pegainfer_comm::raw::fabric_lib::{
    RdmaEngine, TopologyGroup, TransferEngine, TransferEngineBuilder, detect_topology,
};
use pegainfer_comm::raw::p2p_all_to_all::{AllToAllRankHandle, ScalarType};
use pegainfer_comm::{EpBackend, EpBackendParams, EpDtypes, EpRankBuffers, EpTopology};

use crate::config::Config;

const PAGE_SIZE: usize = 4096;

/// Tunables for the EP backend allocation. Centralized here so we can swap
/// in dynamic sizing later.
#[derive(Debug, Clone, Copy)]
pub struct PplxBootstrapParams {
    /// Worst-case tokens any one dispatch on this rank carries. For decode
    /// we set this conservatively to a small batch capacity.
    pub max_num_tokens: usize,
    /// Per-expert padding inside the combine buffer.
    pub expert_padding: usize,
    /// Optional override for `max_private_tokens` (combine-side staging).
    pub max_private_tokens: Option<usize>,
    /// How many NICs to bind per GPU (1 = NVLink-only intra-node).
    pub nets_per_gpu: u8,
    /// Imm-data base distinguishing this all-to-all on the fabric.
    pub imm_base: u32,
}

impl Default for PplxBootstrapParams {
    fn default() -> Self {
        Self {
            max_num_tokens: 8,
            expert_padding: 16,
            max_private_tokens: None,
            nets_per_gpu: 1,
            imm_base: 0x80000000,
        }
    }
}

/// Per-rank scratch the bootstrap owns. Holding these alive for the whole
/// process keeps the device pointers + MR registrations valid for as long
/// as the matching [`EpBackend`] is alive.
#[allow(dead_code)]
pub struct PplxRankResources {
    pub num_routed_host: CudaHostMemory,
    pub send_handle: CUMemAllocHandle,
    pub send_mapping: CUMemMapping,
    pub recv_handle: CUMemAllocHandle,
    pub recv_mapping: CUMemMapping,
    pub sync_handle: CUMemAllocHandle,
    pub sync_mapping: CUMemMapping,
    pub num_routed_mr: MemoryRegionHandle,
    pub num_routed_desc: MemoryRegionDescriptor,
    pub send_buffer_mr: MemoryRegionHandle,
    pub recv_buffer_mr: MemoryRegionHandle,
    pub recv_buffer_desc: MemoryRegionDescriptor,
    pub transfer_engine: Arc<TransferEngine>,
    /// CUMem mappings for *peer* send/recv/sync buffers into this rank's
    /// VA. Kept here so they live as long as the backend; dropping them
    /// would unmap the peer pointers we handed to the kernel.
    pub peer_mappings: Vec<CUMemMapping>,
}

/// Sizing derived from [`PplxBootstrapParams`] + [`Config`].
struct Sizing {
    world_size: usize,
    num_local_experts: usize,
    num_experts_per_token: usize,
    dp_size: usize,
    max_num_tokens: usize,
    max_private_tokens: usize,
    max_recv_tokens: usize,
    hidden_dim: usize,
    hidden_dim_scale: usize,
    in_elemsize: usize,
    out_elemsize: usize,
    scale_elemsize: usize,
    send_buffer_bytes: usize,
    recv_buffer_bytes: usize,
    sync_buffer_bytes: usize,
    num_routed_bytes: usize,
}

/// Cross-rank handle view a Phase-1 thread publishes for the Phase-2
/// threads to peer-map and stitch into `EpRankBuffers`.
#[derive(Clone)]
struct CrossRankView {
    send_handle: CUMemAllocHandle,
    recv_handle: CUMemAllocHandle,
    sync_handle: CUMemAllocHandle,
    send_self_ptr: u64,
    recv_self_ptr: u64,
    sync_self_ptr: u64,
    num_routed_desc: MemoryRegionDescriptor,
    recv_buffer_desc: MemoryRegionDescriptor,
    main_address: DomainAddress,
}

/// What a Phase-1 worker hands back to the main thread.
struct Phase1Output {
    resources: PplxRankResources,
    cross: CrossRankView,
}

/// Build EP backends for all `world_size` ranks living in this process.
///
/// Spawns one worker thread per rank; each thread does `cudaSetDevice` once
/// and owns all CUDA + verbs setup for its rank. The main thread only
/// gathers cross-rank handle descriptors between phases.
pub fn build_intra_node_backends(
    config: &Config,
    devices: &[usize],
    params: PplxBootstrapParams,
) -> Result<(Vec<EpBackend>, Vec<PplxRankResources>)> {
    let world_size = devices.len();
    if world_size == 0 {
        bail!("pplx bootstrap: device list empty");
    }
    if !config.n_routed_experts.is_multiple_of(world_size) {
        bail!(
            "pplx bootstrap: n_routed_experts={} must divide world_size={}",
            config.n_routed_experts,
            world_size
        );
    }

    let sizing = compute_sizing(config, world_size, params);
    let topology_groups = detect_topology().context("pplx bootstrap: detect_topology failed")?;

    // ------------------- Phase 1: local alloc + MR register -------------------
    let mut phase1_slots: Vec<Option<Result<Phase1Output>>> =
        (0..world_size).map(|_| None).collect();
    thread::scope(|s| {
        for ((rank, &dev_ord), slot) in devices.iter().enumerate().zip(phase1_slots.iter_mut()) {
            let sizing = &sizing;
            let topology_groups = &topology_groups;
            let params = params;
            s.spawn(move || {
                *slot = Some(run_phase1(rank, dev_ord, sizing, topology_groups, params));
            });
        }
    });

    let phase1: Vec<Phase1Output> = phase1_slots
        .into_iter()
        .enumerate()
        .map(|(rank, opt)| {
            opt.expect("phase1 worker did not store a result")
                .with_context(|| format!("pplx phase1 rank {rank}"))
        })
        .collect::<Result<_>>()?;

    // Snapshot cross-rank handles for the next scope to read.
    let cross: Arc<Vec<CrossRankView>> = Arc::new(phase1.iter().map(|p| p.cross.clone()).collect());

    // Move local resources into per-rank slots so Phase-2 threads can take
    // ownership without cloning the entire `PplxRankResources` blob.
    let mut local_slots: Vec<Option<PplxRankResources>> =
        phase1.into_iter().map(|p| Some(p.resources)).collect();
    let mut phase2_slots: Vec<Option<Result<(EpBackend, PplxRankResources)>>> =
        (0..world_size).map(|_| None).collect();

    // ------------------- Phase 2: peer-map + EpBackend::new -------------------
    thread::scope(|s| {
        for (((rank, &dev_ord), local_slot), out_slot) in devices
            .iter()
            .enumerate()
            .zip(local_slots.iter_mut())
            .zip(phase2_slots.iter_mut())
        {
            let cross = Arc::clone(&cross);
            let sizing = &sizing;
            let topology_groups = &topology_groups;
            let params = params;
            s.spawn(move || {
                let resources = local_slot
                    .take()
                    .expect("phase2 entered without phase1 resources");
                *out_slot = Some(run_phase2(
                    rank,
                    dev_ord,
                    resources,
                    cross,
                    sizing,
                    topology_groups,
                    params,
                ));
            });
        }
    });

    let mut backends = Vec::with_capacity(world_size);
    let mut resources_out = Vec::with_capacity(world_size);
    for (rank, slot) in phase2_slots.into_iter().enumerate() {
        let (backend, resources) = slot
            .expect("phase2 worker did not store a result")
            .with_context(|| format!("pplx phase2 rank {rank}"))?;
        backends.push(backend);
        resources_out.push(resources);
    }

    Ok((backends, resources_out))
}

fn run_phase1(
    rank: usize,
    dev_ord: usize,
    sizing: &Sizing,
    topology_groups: &[TopologyGroup],
    params: PplxBootstrapParams,
) -> Result<Phase1Output> {
    pegainfer_comm::raw::cuda_lib::rt::cudaSetDevice(dev_ord as i32)
        .with_context(|| format!("cudaSetDevice({dev_ord}) at phase1 entry"))?;

    let dev_id = CudaDeviceId(dev_ord as u8);
    let device = Device::Cuda(dev_id);

    let te = build_te_for(dev_ord, topology_groups, params.nets_per_gpu)?;

    let num_routed_host = CudaHostMemory::alloc(sizing.num_routed_bytes)
        .with_context(|| format!("alloc pinned num_routed host buffer for cuda:{dev_ord}"))?;

    let send_handle = CUMemAllocHandle::new(
        sizing.send_buffer_bytes,
        dev_id,
        CUMemHandleKind::FileDescriptor,
    )
    .with_context(|| format!("CUMem alloc send buffer for cuda:{dev_ord}"))?;
    let send_mapping = send_handle
        .map(device)
        .with_context(|| format!("CUMem map send buffer on cuda:{dev_ord}"))?;

    let recv_handle = CUMemAllocHandle::new(
        sizing.recv_buffer_bytes,
        dev_id,
        CUMemHandleKind::FileDescriptor,
    )
    .with_context(|| format!("CUMem alloc recv buffer for cuda:{dev_ord}"))?;
    let recv_mapping = recv_handle
        .map(device)
        .with_context(|| format!("CUMem map recv buffer on cuda:{dev_ord}"))?;

    let sync_handle = CUMemAllocHandle::new(
        sizing.sync_buffer_bytes,
        dev_id,
        CUMemHandleKind::FileDescriptor,
    )
    .with_context(|| format!("CUMem alloc sync buffer for cuda:{dev_ord}"))?;
    let sync_mapping = sync_handle
        .map(device)
        .with_context(|| format!("CUMem map sync buffer on cuda:{dev_ord}"))?;
    cuda_memset_zero(sync_mapping.data_ptr(), sizing.sync_buffer_bytes)
        .with_context(|| format!("zero sync buffer on cuda:{dev_ord}"))?;

    let (num_routed_mr, num_routed_desc) = te
        .register_memory_allow_remote(num_routed_host.ptr, sizing.num_routed_bytes, Device::Host)
        .with_context(|| format!("register num_routed MR for cuda:{dev_ord}"))?;
    let (send_buffer_mr, _send_desc) = te
        .register_memory_allow_remote(send_mapping.data_ptr(), sizing.send_buffer_bytes, device)
        .with_context(|| format!("register send MR for cuda:{dev_ord}"))?;
    let (recv_buffer_mr, recv_buffer_desc) = te
        .register_memory_allow_remote(recv_mapping.data_ptr(), sizing.recv_buffer_bytes, device)
        .with_context(|| format!("register recv MR for cuda:{dev_ord}"))?;

    let cross = CrossRankView {
        send_handle: send_handle.clone(),
        recv_handle: recv_handle.clone(),
        sync_handle: sync_handle.clone(),
        send_self_ptr: send_mapping.data_ptr().as_ptr() as u64,
        recv_self_ptr: recv_mapping.data_ptr().as_ptr() as u64,
        sync_self_ptr: sync_mapping.data_ptr().as_ptr() as u64,
        num_routed_desc: num_routed_desc.clone(),
        recv_buffer_desc: recv_buffer_desc.clone(),
        main_address: te.main_address(),
    };

    let resources = PplxRankResources {
        num_routed_host,
        send_handle,
        send_mapping,
        recv_handle,
        recv_mapping,
        sync_handle,
        sync_mapping,
        num_routed_mr,
        num_routed_desc,
        send_buffer_mr,
        recv_buffer_mr,
        recv_buffer_desc,
        transfer_engine: te,
        peer_mappings: Vec::new(),
    };

    let _ = rank;
    Ok(Phase1Output { resources, cross })
}

fn run_phase2(
    rank: usize,
    dev_ord: usize,
    mut resources: PplxRankResources,
    cross: Arc<Vec<CrossRankView>>,
    sizing: &Sizing,
    topology_groups: &[TopologyGroup],
    params: PplxBootstrapParams,
) -> Result<(EpBackend, PplxRankResources)> {
    pegainfer_comm::raw::cuda_lib::rt::cudaSetDevice(dev_ord as i32)
        .with_context(|| format!("cudaSetDevice({dev_ord}) at phase2 entry"))?;

    let world_size = cross.len();
    let local_device = Device::Cuda(CudaDeviceId(dev_ord as u8));

    let mut send_ptrs = Vec::with_capacity(world_size);
    let mut recv_ptrs = Vec::with_capacity(world_size);
    let mut sync_ptrs = Vec::with_capacity(world_size);
    let mut peer_mappings = Vec::with_capacity((world_size - 1) * 3);
    for (peer, view) in cross.iter().enumerate() {
        if peer == rank {
            send_ptrs.push(view.send_self_ptr);
            recv_ptrs.push(view.recv_self_ptr);
            sync_ptrs.push(view.sync_self_ptr);
            continue;
        }
        let send_map = view
            .send_handle
            .map(local_device)
            .with_context(|| format!("map peer {peer} send buffer into cuda:{dev_ord} VA"))?;
        let recv_map = view
            .recv_handle
            .map(local_device)
            .with_context(|| format!("map peer {peer} recv buffer into cuda:{dev_ord} VA"))?;
        let sync_map = view
            .sync_handle
            .map(local_device)
            .with_context(|| format!("map peer {peer} sync buffer into cuda:{dev_ord} VA"))?;
        send_ptrs.push(send_map.data_ptr().as_ptr() as u64);
        recv_ptrs.push(recv_map.data_ptr().as_ptr() as u64);
        sync_ptrs.push(sync_map.data_ptr().as_ptr() as u64);
        peer_mappings.push(send_map);
        peer_mappings.push(recv_map);
        peer_mappings.push(sync_map);
    }
    resources.peer_mappings = peer_mappings;

    let group = topology_groups
        .iter()
        .find(|g| g.cuda_device as usize == dev_ord)
        .with_context(|| format!("topology lookup for cuda:{dev_ord}"))?;
    // Pin the a2a worker to a CPU distinct from this rank's TE worker
    // (`cpus[0]`) and UVM watcher (`cpus[2]` / fallback `cpus[1]`); see
    // build_te_for. Without this they all single-CPU-pin to the same
    // core and the kernel-side `tx_ready` spin waits ~50 ms behind the
    // queued worker progress.
    let worker_cpu = group
        .cpus
        .get(3)
        .copied()
        .or_else(|| group.cpus.get(1).copied())
        .or_else(|| group.cpus.first().copied());

    let topology = EpTopology {
        world_size,
        rank,
        node_size: world_size,
        dp_size: sizing.dp_size,
        num_experts: sizing.num_local_experts * world_size,
        num_experts_per_token: sizing.num_experts_per_token,
        hidden_dim: sizing.hidden_dim,
        hidden_dim_scale: sizing.hidden_dim_scale,
        max_num_tokens: sizing.max_num_tokens,
        max_recv_tokens: sizing.max_recv_tokens,
        max_private_tokens: sizing.max_private_tokens,
        expert_padding: params.expert_padding,
    };
    let dtypes = EpDtypes {
        in_elemsize: sizing.in_elemsize,
        out_elemsize: sizing.out_elemsize,
        out_dtype: ScalarType::BF16,
        scale_elemsize: sizing.scale_elemsize,
    };
    let buffers = EpRankBuffers {
        num_routed_ptr: resources.num_routed_host.ptr.as_ptr() as *mut u32,
        num_routed_mr: resources.num_routed_mr,
        send_buffer_ptr: resources.send_mapping.data_ptr().as_ptr(),
        send_buffer_mr: resources.send_buffer_mr,
        recv_buffer_ptr: resources.recv_mapping.data_ptr().as_ptr(),
        recv_buffer_mr: resources.recv_buffer_mr,
        sync_ptrs,
        send_ptrs,
        recv_ptrs,
    };

    let rank_handles: Vec<AllToAllRankHandle> = cross
        .iter()
        .map(|v| {
            AllToAllRankHandle::new(
                v.main_address.clone(),
                v.num_routed_desc.clone(),
                v.recv_buffer_desc.clone(),
            )
        })
        .collect();

    let backend = EpBackend::new(EpBackendParams {
        topology,
        dtypes,
        buffers,
        rank_handles,
        transfer_engine: resources.transfer_engine.clone(),
        device: dev_ord as u8,
        imm_base: params.imm_base,
        worker_cpu,
    })
    .with_context(|| format!("build EpBackend for rank {rank}"))?;

    Ok((backend, resources))
}

fn compute_sizing(config: &Config, world_size: usize, params: PplxBootstrapParams) -> Sizing {
    let num_local_experts = config.n_routed_experts / world_size;
    let num_experts_per_token = config.n_activated_experts;
    let dp_size = 1usize;
    let num_dp_groups = world_size / dp_size;
    let max_num_tokens = params.max_num_tokens;

    let avg_tokens_per_expert = {
        let raw = (max_num_tokens * num_experts_per_token).div_ceil(config.n_routed_experts);
        raw + raw / 5 + 1
    };
    let max_private_tokens = params
        .max_private_tokens
        .unwrap_or(avg_tokens_per_expert * num_local_experts);

    let num_tokens_total = max_num_tokens * num_dp_groups;
    let max_recv_tokens = max_private_tokens * num_dp_groups
        + round_up(
            std::cmp::max(
                std::cmp::min(
                    num_tokens_total * num_experts_per_token
                        + num_local_experts * (params.expert_padding - 1),
                    num_tokens_total * num_local_experts,
                ),
                num_local_experts * params.expert_padding,
            ),
            params.expert_padding,
        );

    let hidden_dim = config.dim;
    let hidden_dim_scale = 0usize;
    let in_elemsize = 2usize;
    let out_elemsize = 2usize;
    let scale_elemsize = 0usize;
    let token_dim_dispatch = round_up(hidden_dim * in_elemsize, 16) + 16;
    let token_dim_combine = round_up(hidden_dim * out_elemsize, 16);
    let token_dim = std::cmp::max(token_dim_dispatch, token_dim_combine);

    let send_buffer_bytes = round_up(max_recv_tokens * token_dim, PAGE_SIZE);
    let recv_buffer_bytes = round_up(max_recv_tokens * token_dim, PAGE_SIZE);
    let sync_buffer_bytes = std::mem::size_of::<u32>() * world_size * 2;
    let num_routed_len = num_dp_groups * config.n_routed_experts;
    let num_routed_bytes = round_up(num_routed_len * std::mem::size_of::<u32>(), PAGE_SIZE);

    Sizing {
        world_size,
        num_local_experts,
        num_experts_per_token,
        dp_size,
        max_num_tokens,
        max_private_tokens,
        max_recv_tokens,
        hidden_dim,
        hidden_dim_scale,
        in_elemsize,
        out_elemsize,
        scale_elemsize,
        send_buffer_bytes,
        recv_buffer_bytes,
        sync_buffer_bytes,
        num_routed_bytes,
    }
}

fn build_te_for(
    dev: usize,
    topology_groups: &[TopologyGroup],
    nets_per_gpu: u8,
) -> Result<Arc<TransferEngine>> {
    let group = topology_groups
        .iter()
        .find(|g| g.cuda_device as usize == dev)
        .with_context(|| format!("pplx bootstrap: cuda:{dev} not in topology"))?;
    if group.cpus.len() < 2 {
        bail!(
            "pplx bootstrap: cuda:{dev} needs >=2 CPUs in topology group; got {}",
            group.cpus.len()
        );
    }
    let worker_cpu = group.cpus[0];
    let uvm_cpu = group.cpus.get(2).copied().unwrap_or(group.cpus[1]);
    let n_doms = std::cmp::min(group.domains.len(), nets_per_gpu as usize);
    let doms = group
        .domains
        .iter()
        .take(n_doms)
        .cloned()
        .collect::<Vec<_>>();
    let mut builder = TransferEngineBuilder::default();
    builder.add_gpu_domains(dev as u8, doms, worker_cpu, uvm_cpu);
    Ok(Arc::new(builder.build().with_context(|| {
        format!("pplx bootstrap: TE build for cuda:{dev}")
    })?))
}

fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}

fn cuda_memset_zero(ptr: NonNull<c_void>, bytes: usize) -> Result<()> {
    let ret =
        unsafe { pegainfer_comm::raw::cuda_lib::cudart_sys::cudaMemset(ptr.as_ptr(), 0, bytes) };
    if ret == 0 {
        Ok(())
    } else {
        bail!("cudaMemset failed with code {ret}");
    }
}
