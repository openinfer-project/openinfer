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
//! NOTE — this is a first cut. It compiles and lays out the right call
//! shape; full functional verification (NVLink P2P + MR registration) is
//! pending live H20 testing.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use pegainfer_comm::raw::cuda_lib::cumem::{
    CUAllocHandle, CUMemAllocHandle, CUMemHandleKind, CUMemMapping,
};
use pegainfer_comm::raw::cuda_lib::{CudaDeviceId, Device};
use pegainfer_comm::raw::fabric_lib::api::{MemoryRegionDescriptor, MemoryRegionHandle};
use pegainfer_comm::raw::fabric_lib::{
    RdmaEngine, TransferEngine, TransferEngineBuilder, detect_topology,
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
    pub num_routed_host: Vec<u32>, // pinned host buffer (Vec is page-aligned + locked downstream)
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
}

/// Build EP backends for all `world_size` ranks living in this process.
///
/// Caller is responsible for keeping the returned `_resources` alive — drop
/// them only after all `EpBackend`s have been dropped.
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
    let num_local_experts = config.n_routed_experts / world_size;

    let max_num_tokens = params.max_num_tokens;
    let num_experts_per_token = config.n_activated_experts;
    let num_dp_groups = 1usize; // pure EP, no DP

    let avg_tokens_per_expert = {
        let raw = (max_num_tokens * num_experts_per_token).div_ceil(config.n_routed_experts);
        // Match upstream Python: ceil_div * 1.2, but stay integer.
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

    // Dispatch wire layout: BF16 hidden, no FP8 scale.
    let hidden_dim = config.dim;
    let hidden_dim_scale = 0usize;
    let in_elemsize = 2usize; // BF16
    let out_elemsize = 2usize; // BF16
    let scale_elemsize = 0usize;
    let token_dim_dispatch = round_up(hidden_dim * in_elemsize, 16) + 16;
    let token_dim_combine = round_up(hidden_dim * out_elemsize, 16);
    let token_dim = std::cmp::max(token_dim_dispatch, token_dim_combine);

    let send_buffer_bytes = round_up(max_recv_tokens * token_dim, PAGE_SIZE);
    let recv_buffer_bytes = round_up(max_recv_tokens * token_dim, PAGE_SIZE);
    let sync_buffer_bytes = std::mem::size_of::<u32>() * world_size * 2;

    // Build one TransferEngine *per rank*, each binding to its own GPU+NIC.
    // pplx-garden's AllToAllRankHandle carries a single `address` (the
    // peer's main NIC). With a single shared TE, `te.main_address()` only
    // returns worker[0]'s NIC, so all 8 RankHandles end up pointing at the
    // same NIC and RDMA writes target the wrong domain (local protection
    // error). One TE per GPU gives each rank its own NIC main address.
    let system_topo = detect_topology().context("pplx bootstrap: detect_topology failed")?;

    let build_te_for = |dev: usize| -> Result<Arc<TransferEngine>> {
        let group = system_topo
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
        let n_doms = std::cmp::min(group.domains.len(), params.nets_per_gpu as usize);
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
    };

    // Phase 1: per-rank allocation + MR registration in this rank's own TE.
    let mut resources: Vec<PplxRankResources> = Vec::with_capacity(world_size);
    for &dev_ord in devices {
        let dev_id = CudaDeviceId(dev_ord as u8);
        let device = Device::Cuda(dev_id);

        let te = build_te_for(dev_ord)?;

        let num_routed_len = num_dp_groups * config.n_routed_experts;
        let num_routed_host = vec![0u32; num_routed_len];

        let send_handle =
            CUMemAllocHandle::new(send_buffer_bytes, dev_id, CUMemHandleKind::FileDescriptor)
                .with_context(|| format!("CUMem alloc send buffer for cuda:{dev_ord}"))?;
        let send_mapping = send_handle
            .map(device)
            .with_context(|| format!("CUMem map send buffer on cuda:{dev_ord}"))?;

        let recv_handle =
            CUMemAllocHandle::new(recv_buffer_bytes, dev_id, CUMemHandleKind::FileDescriptor)
                .with_context(|| format!("CUMem alloc recv buffer for cuda:{dev_ord}"))?;
        let recv_mapping = recv_handle
            .map(device)
            .with_context(|| format!("CUMem map recv buffer on cuda:{dev_ord}"))?;

        let sync_handle =
            CUMemAllocHandle::new(sync_buffer_bytes, dev_id, CUMemHandleKind::FileDescriptor)
                .with_context(|| format!("CUMem alloc sync buffer for cuda:{dev_ord}"))?;
        let sync_mapping = sync_handle
            .map(device)
            .with_context(|| format!("CUMem map sync buffer on cuda:{dev_ord}"))?;

        let (num_routed_mr, num_routed_desc) = te
            .register_memory_allow_remote(
                NonNull::new(num_routed_host.as_ptr() as *mut c_void)
                    .expect("num_routed buffer non-null"),
                num_routed_len * std::mem::size_of::<u32>(),
                Device::Host,
            )
            .with_context(|| format!("register num_routed MR for cuda:{dev_ord}"))?;
        let (send_buffer_mr, _send_desc) = te
            .register_memory_allow_remote(send_mapping.data_ptr(), send_buffer_bytes, device)
            .with_context(|| format!("register send MR for cuda:{dev_ord}"))?;
        let (recv_buffer_mr, recv_buffer_desc) = te
            .register_memory_allow_remote(recv_mapping.data_ptr(), recv_buffer_bytes, device)
            .with_context(|| format!("register recv MR for cuda:{dev_ord}"))?;

        resources.push(PplxRankResources {
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
        });
    }

    // Phase 2: capture per-rank handle descriptors + peer NIC addresses.
    let rank_addresses: Vec<_> = resources
        .iter()
        .map(|r| r.transfer_engine.main_address())
        .collect();
    let rank_handle_parts: Vec<(MemoryRegionDescriptor, MemoryRegionDescriptor)> = (0..world_size)
        .map(|r| {
            (
                resources[r].num_routed_desc.clone(),
                resources[r].recv_buffer_desc.clone(),
            )
        })
        .collect();

    // For each rank, build (send_ptrs, recv_ptrs, sync_ptrs) of length
    // world_size by mapping each peer's CUMem alloc into this rank's
    // device VA. The self-entries are the rank's own mapping.
    let mut peer_send_ptrs: Vec<Vec<u64>> = Vec::with_capacity(world_size);
    let mut peer_recv_ptrs: Vec<Vec<u64>> = Vec::with_capacity(world_size);
    let mut peer_sync_ptrs: Vec<Vec<u64>> = Vec::with_capacity(world_size);
    // Keep the peer mappings alive for the lifetime of the backend.
    let mut peer_mapping_holders: Vec<Vec<CUMemMapping>> = Vec::with_capacity(world_size);

    for (rank, &dev_ord) in devices.iter().enumerate() {
        let local_device = Device::Cuda(CudaDeviceId(dev_ord as u8));
        let mut send_v = Vec::with_capacity(world_size);
        let mut recv_v = Vec::with_capacity(world_size);
        let mut sync_v = Vec::with_capacity(world_size);
        let mut holders = Vec::with_capacity(world_size);
        for peer in 0..world_size {
            if peer == rank {
                send_v.push(resources[peer].send_mapping.data_ptr().as_ptr() as u64);
                recv_v.push(resources[peer].recv_mapping.data_ptr().as_ptr() as u64);
                sync_v.push(resources[peer].sync_mapping.data_ptr().as_ptr() as u64);
            } else {
                let send_map =
                    resources[peer]
                        .send_handle
                        .map(local_device)
                        .with_context(|| {
                            format!("map peer {peer} send buffer into cuda:{dev_ord} VA")
                        })?;
                let recv_map =
                    resources[peer]
                        .recv_handle
                        .map(local_device)
                        .with_context(|| {
                            format!("map peer {peer} recv buffer into cuda:{dev_ord} VA")
                        })?;
                let sync_map =
                    resources[peer]
                        .sync_handle
                        .map(local_device)
                        .with_context(|| {
                            format!("map peer {peer} sync buffer into cuda:{dev_ord} VA")
                        })?;
                send_v.push(send_map.data_ptr().as_ptr() as u64);
                recv_v.push(recv_map.data_ptr().as_ptr() as u64);
                sync_v.push(sync_map.data_ptr().as_ptr() as u64);
                holders.push(send_map);
                holders.push(recv_map);
                holders.push(sync_map);
            }
        }
        peer_send_ptrs.push(send_v);
        peer_recv_ptrs.push(recv_v);
        peer_sync_ptrs.push(sync_v);
        peer_mapping_holders.push(holders);
    }

    // Phase 3: construct EpBackends.
    let mut backends = Vec::with_capacity(world_size);
    for (rank, &dev_ord) in devices.iter().enumerate() {
        let group = system_topo
            .iter()
            .find(|g| g.cuda_device as usize == dev_ord)
            .expect("topology lookup already validated");
        let worker_cpu = group.cpus.first().copied();

        let topology = EpTopology {
            world_size,
            rank,
            node_size: world_size,
            dp_size: num_dp_groups,
            num_experts: config.n_routed_experts,
            num_experts_per_token,
            hidden_dim,
            hidden_dim_scale,
            max_num_tokens,
            max_recv_tokens,
            max_private_tokens,
            expert_padding: params.expert_padding,
        };
        let dtypes = EpDtypes {
            in_elemsize,
            out_elemsize,
            out_dtype: ScalarType::BF16,
            scale_elemsize,
        };
        let buffers = EpRankBuffers {
            num_routed_ptr: resources[rank].num_routed_host.as_ptr() as *mut u32,
            num_routed_mr: resources[rank].num_routed_mr,
            send_buffer_ptr: resources[rank].send_mapping.data_ptr().as_ptr(),
            send_buffer_mr: resources[rank].send_buffer_mr,
            recv_buffer_ptr: resources[rank].recv_mapping.data_ptr().as_ptr(),
            recv_buffer_mr: resources[rank].recv_buffer_mr,
            sync_ptrs: peer_sync_ptrs[rank].clone(),
            send_ptrs: peer_send_ptrs[rank].clone(),
            recv_ptrs: peer_recv_ptrs[rank].clone(),
        };

        let rank_handles: Vec<AllToAllRankHandle> = rank_handle_parts
            .iter()
            .enumerate()
            .map(|(peer, (nr, rb))| {
                AllToAllRankHandle::new(rank_addresses[peer].clone(), nr.clone(), rb.clone())
            })
            .collect();

        let backend = EpBackend::new(EpBackendParams {
            topology,
            dtypes,
            buffers,
            rank_handles,
            transfer_engine: resources[rank].transfer_engine.clone(),
            device: dev_ord as u8,
            imm_base: params.imm_base,
            worker_cpu,
        })
        .with_context(|| format!("build EpBackend for rank {rank}"))?;
        backends.push(backend);
    }

    // Leak the peer mapping holders into the resources (last entry — we
    // attach them to each rank's record so they live as long as the
    // backend does).
    for (rank, holders) in peer_mapping_holders.into_iter().enumerate() {
        // The peer mappings are not owned by the bootstrap rank's
        // PplxRankResources struct fields; stash them in a side vec to
        // keep alive. We rebuild a fresh Vec stashed alongside.
        // The simplest correctness path: leak via Box::leak so they live
        // for process lifetime. Bootstrap is one-shot, so leak is fine.
        let leaked: &'static [CUMemMapping] = Box::leak(holders.into_boxed_slice());
        std::mem::forget(leaked); // belt-and-suspenders
        let _ = rank;
    }

    Ok((backends, resources))
}

fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}
