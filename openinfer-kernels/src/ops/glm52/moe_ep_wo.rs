//! GLM5.2 EP4 weight-only routed-expert chain: tile metadata + masked
//! grouped bf16×fp8 mma GEMM + weighted SiLU, all on the DeepEP aligned
//! receive layout. The Blackwell-native replacement for the sm_90a DeepGEMM
//! masked grouped chain — no fp8 activation re-quant, no masked relayout,
//! W2 writes straight into the aligned slots `decode_combine` addresses.
//! See `csrc/glm52/glm52_moe_ep_wo.cu`.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::deepgemm_grouped::Glm52DeepGemmGroupedFp8Kind;
use crate::ffi;
use crate::tensor::DeviceContext;

/// Aligned rows per work tile (8 | 64 = the DeepEP expert alignment, so a
/// tile never straddles an expert segment).
pub const GLM52_MOE_EP_WO_TILE_ROWS: usize = 8;

/// Host-side worst-case tile count for a step: every expert can open one
/// partial tile, plus one tile per full `TILE_ROWS` rows of the global
/// expanded budget (each source token contributes at most one row per
/// expert, so per-expert rows are bounded by `global_tokens` and total rows
/// by `global_tokens * topk`).
#[must_use]
pub fn glm52_moe_ep_wo_max_tiles(groups: usize, global_tokens: usize, topk: usize) -> usize {
    groups + (global_tokens * topk).div_ceil(GLM52_MOE_EP_WO_TILE_ROWS)
}

/// psum_expert (i32 aligned running ends from the DeepEP dispatch) → compact
/// tile work list (`tiles` holds int2 entries — pass an i32 buffer of len
/// `2 * max_tiles` — plus a device tile count). `m_capacity` is the
/// host-derived aligned-row bound and `row_cap` the per-expert row bound
/// (the step's `global_tokens`); the kernel device-traps if any segment
/// violates them — a cross-rank token-count disagreement.
pub fn glm52_moe_ep_wo_tiles_launch(
    ctx: &DeviceContext,
    groups: usize,
    m_capacity: usize,
    row_cap: usize,
    max_tiles: usize,
    psum_expert: &CudaSlice<i32>,
    tiles: &mut CudaSlice<i32>,
    tile_count: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0 && m_capacity > 0 && row_cap > 0 && max_tiles > 0,
        "GLM5.2 EP-WO tiles needs positive groups/m_capacity/row_cap/max_tiles, got \
         groups={groups}, m_capacity={m_capacity}, row_cap={row_cap}, max_tiles={max_tiles}"
    );
    ensure!(
        psum_expert.len() >= groups && tiles.len() >= 2 * max_tiles && !tile_count.is_empty(),
        "GLM5.2 EP-WO tiles buffers too small for {groups} groups / {max_tiles} tiles: \
         psum={}, tiles={}, tile_count={}",
        psum_expert.len(),
        tiles.len(),
        tile_count.len()
    );
    let (psum_ptr, _psum_guard) = psum_expert.device_ptr(&ctx.stream);
    let (tiles_ptr, _tiles_guard) = tiles.device_ptr_mut(&ctx.stream);
    let (count_ptr, _count_guard) = tile_count.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_moe_ep_wo_tiles_cuda(
            psum_ptr as *const i32,
            tiles_ptr as *mut i32,
            count_ptr as *mut i32,
            groups as i32,
            m_capacity as i32,
            row_cap as i32,
            max_tiles as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 EP-WO tiles launch failed: {err}"))
}

/// Masked grouped weight-only GEMM over the tile list: for every tile,
/// `out[rows, :n] = activation[rows, :k] (bf16) @ deq(weight[expert])`, f32
/// accumulation in a fixed per-shape order. Activation and output rows are
/// the DeepEP aligned receive slots; the weight/scale banks are the packed
/// per-rank expert slabs (`[groups, n, k]` e4m3 / `[groups, n/128, k/128]`
/// f32, checkpoint layout as-is). `row_weights` (per aligned row, the
/// dispatch route weights on W2) scales the f32 accumulator before the bf16
/// store. The grid is shaped at `max_tiles`; blocks past the device tile
/// count retire immediately (CUDA-graph stable).
#[allow(clippy::too_many_arguments)]
pub fn glm52_moe_ep_wo_masked_mma_launch(
    ctx: &DeviceContext,
    kind: Glm52DeepGemmGroupedFp8Kind,
    groups: usize,
    max_tiles: usize,
    activation: &CudaSlice<bf16>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    tiles: &CudaSlice<i32>,
    tile_count: &CudaSlice<i32>,
    row_weights: Option<&CudaSlice<f32>>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let (n, k) = kind.shape();
    ensure!(
        groups > 0 && max_tiles > 0,
        "GLM5.2 EP-WO masked mma needs positive groups/max_tiles, got {groups}/{max_tiles}"
    );
    ensure!(
        weight.len() >= groups * n * k
            && weight_scale.len() >= groups * (n / 128) * (k / 128)
            && tiles.len() >= 2 * max_tiles
            && !tile_count.is_empty()
            && activation.len() >= k
            && out.len() >= n
            && activation.len() / k >= out.len() / n,
        "GLM5.2 EP-WO masked mma {kind:?} buffers too small: act {}, w {}, w_scale {}, \
         tiles {}, out {}",
        activation.len(),
        weight.len(),
        weight_scale.len(),
        tiles.len(),
        out.len()
    );
    if let Some(weights) = row_weights {
        ensure!(
            weights.len() >= out.len() / n,
            "GLM5.2 EP-WO masked mma {kind:?} row weights smaller than the output rows: {}",
            weights.len()
        );
    }
    let (act_ptr, _act_guard) = activation.device_ptr(&ctx.stream);
    let (w_ptr, _w_guard) = weight.device_ptr(&ctx.stream);
    let (w_scale_ptr, _w_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (tiles_ptr, _tiles_guard) = tiles.device_ptr(&ctx.stream);
    let (count_ptr, _count_guard) = tile_count.device_ptr(&ctx.stream);
    let row_weights_guarded = row_weights.map(|w| w.device_ptr(&ctx.stream));
    let rw_ptr = row_weights_guarded
        .as_ref()
        .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const f32);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_moe_ep_wo_masked_mma_cuda(
            act_ptr as *const ffi::Half,
            w_ptr as *const u8,
            w_scale_ptr as *const f32,
            tiles_ptr as *const i32,
            count_ptr as *const i32,
            rw_ptr,
            out_ptr as *mut ffi::Half,
            n as i32,
            k as i32,
            max_tiles as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 EP-WO masked mma {kind:?} launch failed: {err}"))
}

/// `silu(gate) * up` over the tile rows, bf16 out (the route weight applies
/// to the f32 W2 output instead — see the masked mma's `row_weights`).
/// `input` rows are the W13 gate|up outputs (`[·, 2*inter]`), `output` the
/// W2 activation rows (`[·, inter]`) — all in the aligned receive layout.
pub fn glm52_moe_ep_wo_silu_launch(
    ctx: &DeviceContext,
    inter: usize,
    max_tiles: usize,
    input: &CudaSlice<bf16>,
    tiles: &CudaSlice<i32>,
    tile_count: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        inter > 0 && max_tiles > 0,
        "GLM5.2 EP-WO SiLU needs positive inter/max_tiles, got {inter}/{max_tiles}"
    );
    ensure!(
        input.len() >= 2 * inter
            && output.len() >= inter
            && input.len() / (2 * inter) >= output.len() / inter
            && tiles.len() >= 2 * max_tiles
            && !tile_count.is_empty(),
        "GLM5.2 EP-WO SiLU buffers too small: input {}, tiles {}, output {}",
        input.len(),
        tiles.len(),
        output.len()
    );
    let (in_ptr, _in_guard) = input.device_ptr(&ctx.stream);
    let (tiles_ptr, _tiles_guard) = tiles.device_ptr(&ctx.stream);
    let (count_ptr, _count_guard) = tile_count.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_moe_ep_wo_silu_cuda(
            in_ptr as *const ffi::Half,
            tiles_ptr as *const i32,
            count_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            inter as i32,
            max_tiles as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 EP-WO SiLU launch failed: {err}"))
}
