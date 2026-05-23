//! CUMem / fabric / peer-map bootstrap for the pplx-garden EP backend.
//!
//! # Threading model
//!
//! Each rank gets its own worker thread for the whole bootstrap. Each
//! thread calls `cudaSetDevice` once at entry, so every CUDA call inside
//! (CUMem alloc, host alloc, MR register, peer map, EpBackend::new) runs
//! against the correct device context.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result, bail, ensure};
use pegainfer_core::cpu_topology::{CpuId, RankThreadPlacementPlan};

use crate::raw::cuda_lib::cumem::{
    CUAllocHandle, CUMemAllocHandle, CUMemHandleKind, CUMemMapping,
};
use crate::raw::cuda_lib::{CudaDeviceId, CudaHostMemory, Device};
use crate::raw::fabric_lib::api::{
    DomainAddress, MemoryRegionDescriptor, MemoryRegionHandle,
};
use crate::raw::fabric_lib::{
    RdmaEngine, TopologyGroup, TransferEngine, TransferEngineBuilder, detect_topology,
};
use crate::raw::p2p_all_to_all::{AllToAllRankHandle, ScalarType};
use crate::{EpBackend, EpBackendParams, EpDtypes, EpRankBuffers, EpTopology};

const PAGE_SIZE: usize = 4096;

/// Model-level parameters the EP bootstrap needs for buffer sizing.
#[derive(Debug, Clone, Copy)]
pub struct EpModelShape {
    /// Total number of routed experts across all ranks.
    pub n_routed_experts: usize,
    /// Number of experts each token is routed to (top-k).
    pub n_activated_experts: usize,
    /// Hidden dimension of the token tensors.
    pub hidden_dim: usize,
}

/// Tunables for the EP backend allocation.
#[derive(Debug, Clone, Copy)]
pub struct PplxBootstrapParams {
    /// Worst-case tokens any one dispatch on this rank carries.
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
    num_routed_host: CudaHostMemory,
    send_handle: CUMemAllocHandle,
    send_mapping: CUMemMapping,
    recv_handle: CUMemAllocHandle,
    recv_mapping: CUMemMapping,
    sync_handle: CUMemAllocHandle,
    sync_mapping: CUMemMapping,
    num_routed_mr: MemoryRegionHandle,
    num_routed_desc: MemoryRegionDescriptor,
    send_buffer_mr: MemoryRegionHandle,
    recv_buffer_mr: MemoryRegionHandle,
    recv_buffer_desc: MemoryRegionDescriptor,
    transfer_engine: Arc<TransferEngine>,
    peer_mappings: Vec<CUMemMapping>,
}

struct Sizing {
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

struct Phase1Output {
    resources: PplxRankResources,
    cross: CrossRankView,
}

#[derive(Clone, Debug)]
struct PplxRankThreadPlacement {
    rank: usize,
    device_ordinal: usize,
    numa_node: usize,
    cpu_slice: String,
    rank_worker_cpu: CpuId,
    a2a_worker_cpu: CpuId,
    te_worker_cpu: CpuId,
    uvm_worker_cpu: CpuId,
}

struct PplxThreadPlacementPlan {
    ranks: Vec<PplxRankThreadPlacement>,
}

impl PplxThreadPlacementPlan {
    fn new(
        devices: &[usize],
        rank_worker_placement: &RankThreadPlacementPlan,
    ) -> Result<Self> {
        let mut ranks = Vec::with_capacity(devices.len());
        for (rank, &device_ordinal) in devices.iter().enumerate() {
            let rank_placement = rank_worker_placement.rank(rank)?;
            ensure!(
                rank_placement.device_ordinal == device_ordinal,
                "pplx placement mismatch for rank {rank}: devices has cuda:{device_ordinal}, \
                 rank worker placement has cuda:{}",
                rank_placement.device_ordinal,
            );
            let te_worker_cpu = rank_placement.role_cpu(1, "TE worker")?;
            let a2a_worker_cpu = rank_placement.role_cpu(2, "a2a worker")?;
            let uvm_worker_cpu = rank_placement.role_cpu(3, "UVM worker")?;

            ranks.push(PplxRankThreadPlacement {
                rank,
                device_ordinal,
                numa_node: rank_placement.numa_node,
                cpu_slice: rank_placement.cpu_slice_display(),
                rank_worker_cpu: rank_placement.rank_worker_cpu,
                a2a_worker_cpu,
                te_worker_cpu,
                uvm_worker_cpu,
            });
        }
        Ok(Self { ranks })
    }

    fn rank(&self, rank: usize) -> Result<PplxRankThreadPlacement> {
        self.ranks
            .get(rank)
            .cloned()
            .with_context(|| format!("missing pplx thread placement for rank {rank}"))
    }

    fn emit_startup_info(&self) {
        for p in &self.ranks {
            log::info!(
                "pplx thread placement rank={} cuda:{} numa={} cpu_slice={} \
                 rank_worker_cpu={} te_worker_cpu={} a2a_worker_cpu={} uvm_worker_cpu={}",
                p.rank,
                p.device_ordinal,
                p.numa_node,
                p.cpu_slice,
                p.rank_worker_cpu,
                p.te_worker_cpu,
                p.a2a_worker_cpu,
                p.uvm_worker_cpu,
            );
        }
    }
}

/// Build EP backends for all ranks with auto-detected thread placement.
pub fn build_intra_node_backends_for_devices(
    shape: EpModelShape,
    devices: &[usize],
    params: PplxBootstrapParams,
) -> Result<(Vec<EpBackend>, Vec<PplxRankResources>)> {
    let rank_worker_placement = RankThreadPlacementPlan::for_devices(devices)?;
    build_intra_node_backends(shape, devices, &rank_worker_placement, params)
}

/// Build EP backends for all ranks with a caller-provided thread placement.
pub fn build_intra_node_backends(
    shape: EpModelShape,
    devices: &[usize],
    rank_worker_placement: &RankThreadPlacementPlan,
    params: PplxBootstrapParams,
) -> Result<(Vec<EpBackend>, Vec<PplxRankResources>)> {
    let world_size = devices.len();
    if world_size == 0 {
        bail!("pplx bootstrap: device list empty");
    }
    if !shape.n_routed_experts.is_multiple_of(world_size) {
        bail!(
            "pplx bootstrap: n_routed_experts={} must divide world_size={}",
            shape.n_routed_experts,
            world_size
        );
    }

    let sizing = compute_sizing(shape, world_size, params);
    let topology_groups =
        detect_topology().context("pplx bootstrap: detect_topology failed")?;
    let thread_placement =
        PplxThreadPlacementPlan::new(devices, rank_worker_placement)?;
    thread_placement.emit_startup_info();

    // ------------------- Phase 1: local alloc + MR register -------------------
    let mut phase1_slots: Vec<Option<Result<Phase1Output>>> =
        (0..world_size).map(|_| None).collect();
    thread::scope(|s| {
        for ((rank, &dev_ord), slot) in
            devices.iter().enumerate().zip(phase1_slots.iter_mut())
        {
            let sizing = &sizing;
            let topology_groups = &topology_groups;
            let placement = thread_placement.rank(rank);
            let params = params;
            s.spawn(move || {
                *slot = Some(placement.and_then(|placement| {
                    run_phase1(
                        rank,
                        dev_ord,
                        sizing,
                        topology_groups,
                        placement,
                        params,
                    )
                }));
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

    let cross: Arc<Vec<CrossRankView>> =
        Arc::new(phase1.iter().map(|p| p.cross.clone()).collect());
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
            let placement = thread_placement.rank(rank);
            let params = params;
            s.spawn(move || {
                let resources =
                    local_slot.take().expect("phase2 entered without phase1 resources");
                *out_slot = Some(placement.and_then(|placement| {
                    run_phase2(
                        rank, dev_ord, resources, cross, sizing, placement, params,
                    )
                }));
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
    placement: PplxRankThreadPlacement,
    params: PplxBootstrapParams,
) -> Result<Phase1Output> {
    crate::raw::cuda_lib::rt::cudaSetDevice(dev_ord as i32)
        .with_context(|| format!("cudaSetDevice({dev_ord}) at phase1 entry"))?;

    let dev_id = CudaDeviceId(dev_ord as u8);
    let device = Device::Cuda(dev_id);

    let te = build_te_for(dev_ord, topology_groups, placement, params.nets_per_gpu)?;

    let num_routed_host =
        CudaHostMemory::alloc(sizing.num_routed_bytes).with_context(|| {
            format!("alloc pinned num_routed host buffer for cuda:{dev_ord}")
        })?;

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
        .register_memory_allow_remote(
            num_routed_host.ptr,
            sizing.num_routed_bytes,
            Device::Host,
        )
        .with_context(|| format!("register num_routed MR for cuda:{dev_ord}"))?;
    let (send_buffer_mr, _send_desc) = te
        .register_memory_allow_remote(
            send_mapping.data_ptr(),
            sizing.send_buffer_bytes,
            device,
        )
        .with_context(|| format!("register send MR for cuda:{dev_ord}"))?;
    let (recv_buffer_mr, recv_buffer_desc) = te
        .register_memory_allow_remote(
            recv_mapping.data_ptr(),
            sizing.recv_buffer_bytes,
            device,
        )
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
    placement: PplxRankThreadPlacement,
    params: PplxBootstrapParams,
) -> Result<(EpBackend, PplxRankResources)> {
    crate::raw::cuda_lib::rt::cudaSetDevice(dev_ord as i32)
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
        let send_map = view.send_handle.map(local_device).with_context(|| {
            format!("map peer {peer} send buffer into cuda:{dev_ord} VA")
        })?;
        let recv_map = view.recv_handle.map(local_device).with_context(|| {
            format!("map peer {peer} recv buffer into cuda:{dev_ord} VA")
        })?;
        let sync_map = view.sync_handle.map(local_device).with_context(|| {
            format!("map peer {peer} sync buffer into cuda:{dev_ord} VA")
        })?;
        send_ptrs.push(send_map.data_ptr().as_ptr() as u64);
        recv_ptrs.push(recv_map.data_ptr().as_ptr() as u64);
        sync_ptrs.push(sync_map.data_ptr().as_ptr() as u64);
        peer_mappings.push(send_map);
        peer_mappings.push(recv_map);
        peer_mappings.push(sync_map);
    }
    resources.peer_mappings = peer_mappings;

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
        worker_cpu: Some(placement.a2a_worker_cpu.as_u16()),
    })
    .with_context(|| format!("build EpBackend for rank {rank}"))?;

    Ok((backend, resources))
}

fn compute_sizing(
    shape: EpModelShape,
    world_size: usize,
    params: PplxBootstrapParams,
) -> Sizing {
    let num_local_experts = shape.n_routed_experts / world_size;
    let num_experts_per_token = shape.n_activated_experts;
    let dp_size = 1usize;
    let num_dp_groups = world_size / dp_size;
    let max_num_tokens = params.max_num_tokens;

    let avg_tokens_per_expert = {
        let raw =
            (max_num_tokens * num_experts_per_token).div_ceil(shape.n_routed_experts);
        raw + raw / 5 + 1
    };
    let max_private_tokens =
        params.max_private_tokens.unwrap_or(avg_tokens_per_expert * num_local_experts);

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

    let hidden_dim = shape.hidden_dim;
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
    let num_routed_len = num_dp_groups * shape.n_routed_experts;
    let num_routed_bytes =
        round_up(num_routed_len * std::mem::size_of::<u32>(), PAGE_SIZE);

    Sizing {
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
    placement: PplxRankThreadPlacement,
    nets_per_gpu: u8,
) -> Result<Arc<TransferEngine>> {
    let group = topology_groups
        .iter()
        .find(|g| g.cuda_device as usize == dev)
        .with_context(|| format!("pplx bootstrap: cuda:{dev} not in topology"))?;
    let n_doms = std::cmp::min(group.domains.len(), nets_per_gpu as usize);
    let doms = group.domains.iter().take(n_doms).cloned().collect::<Vec<_>>();
    let mut builder = TransferEngineBuilder::default();
    builder.add_gpu_domains(
        dev as u8,
        doms,
        placement.te_worker_cpu.as_u16(),
        placement.uvm_worker_cpu.as_u16(),
    );
    Ok(Arc::new(
        builder
            .build()
            .with_context(|| format!("pplx bootstrap: TE build for cuda:{dev}"))?,
    ))
}

fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 { value } else { value.div_ceil(multiple) * multiple }
}

fn cuda_memset_zero(ptr: NonNull<c_void>, bytes: usize) -> Result<()> {
    let ret =
        unsafe { crate::raw::cuda_lib::cudart_sys::cudaMemset(ptr.as_ptr(), 0, bytes) };
    if ret == 0 {
        Ok(())
    } else {
        bail!("cudaMemset failed with code {ret}");
    }
}
