//! GLM5.2 EP8 routed-expert GEMM: DeepGEMM SM90 MGroupedMasked fp8
//! blockscale, AOT-instantiated (no JIT, no torch). The metadata kernel
//! bridges the DeepEP aligned-segment recv layout to the masked layout via
//! `masked_m` + `row_map`; the remap kernel puts the W2 output back into the
//! aligned slots `decode_combine` addresses. See
//! `csrc/glm52/glm52_deepgemm_grouped.cu`.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

const GLM52_DEEPGEMM_GROUPED_W13_KIND: i32 = 1;
const GLM52_DEEPGEMM_GROUPED_W2_KIND: i32 = 2;
/// Per-expert row alignment of the DeepEP recv segment layout (a fixed design
/// constant shared with the vendored shim).
pub const GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT: usize = 64;
/// Masked-layout constants baked into the AOT GEMM instantiation: one rank's
/// local expert count and the per-expert row capacity (the DP8 protocol's
/// worst case — all 64 global tokens routing one row each to one expert).
pub const GLM52_DEEPGEMM_MASKED_GROUPS: usize = 32;
pub const GLM52_DEEPGEMM_MASKED_CAP: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52DeepGemmGroupedFp8Kind {
    W13,
    W2,
}

impl Glm52DeepGemmGroupedFp8Kind {
    fn abi(self) -> i32 {
        match self {
            Self::W13 => GLM52_DEEPGEMM_GROUPED_W13_KIND,
            Self::W2 => GLM52_DEEPGEMM_GROUPED_W2_KIND,
        }
    }

    /// The operand's `(n, k)` (shared with the EP-WO weight-only chain).
    pub fn shape(self) -> (usize, usize) {
        match self {
            Self::W13 => (4096, 6144),
            Self::W2 => (6144, 2048),
        }
    }
}

/// psum (i32 aligned running ends) → aligned segment starts (`expert_offsets`,
/// with `[groups]` = the aligned end), per-expert real row counts
/// (`masked_m`), and the aligned-row → masked-slot map (`row_map`, -1 across
/// alignment gaps). `m_capacity` is the row bound the caller's quant kernels
/// cover; the kernel device-traps if any segment ends past it (a cross-rank
/// token-count disagreement) or exceeds the masked per-expert capacity.
pub fn glm52_deepgemm_grouped_fp8_metadata_launch(
    ctx: &DeviceContext,
    groups: usize,
    m_capacity: usize,
    psum_expert: &CudaSlice<i32>,
    expert_offsets: &mut CudaSlice<i64>,
    masked_m: &mut CudaSlice<i32>,
    row_map: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0 && m_capacity > 0,
        "GLM5.2 DeepGEMM grouped FP8 metadata needs groups>0 and m_capacity>0, got groups={groups}, m_capacity={m_capacity}"
    );
    ensure!(
        psum_expert.len() >= groups
            && expert_offsets.len() > groups
            && masked_m.len() >= groups
            && row_map.len() >= m_capacity,
        "GLM5.2 DeepGEMM grouped FP8 metadata buffers too small for {groups} groups / {m_capacity} rows: psum={}, offsets={}, masked_m={}, row_map={}",
        psum_expert.len(),
        expert_offsets.len(),
        masked_m.len(),
        row_map.len()
    );
    let (psum_ptr, _psum_guard) = psum_expert.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr_mut(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr_mut(&ctx.stream);
    let (map_ptr, _map_guard) = row_map.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_grouped_fp8_metadata_cuda(
            psum_ptr as *const i32,
            offsets_ptr as *mut i64,
            masked_ptr as *mut i32,
            map_ptr as *mut i32,
            groups as i32,
            m_capacity as i32,
            GLM52_DEEPGEMM_GROUPED_EXPERT_ALIGNMENT as i32,
            GLM52_DEEPGEMM_MASKED_CAP as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM grouped FP8 metadata launch failed: {err}"))
}

/// Masked grouped fp8 GEMM over the rank's 32 local experts:
/// `out[g, :masked_m[g], n] = deq(weight[g]) @ deq(activation[g])`.
/// Activation `[32, 64, k]` fp8, activation scale `[32, k/128, 64]` f32
/// mn-major, weight `[32, n, k]` fp8 (bank layout as-is), weight scale
/// `[32, n/128, k/128]` f32 (checkpoint layout as-is), out `[32, 64, n]`
/// bf16. Requires sm_90a (NOT_SUPPORTED elsewhere).
pub fn glm52_deepgemm_masked_grouped_fp8_launch(
    ctx: &DeviceContext,
    kind: Glm52DeepGemmGroupedFp8Kind,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    masked_m: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    let (n, k) = kind.shape();
    let groups = GLM52_DEEPGEMM_MASKED_GROUPS;
    let cap = GLM52_DEEPGEMM_MASKED_CAP;
    ensure!(
        activation.len() >= groups * cap * k
            && activation_scale.len() >= groups * (k / 128) * cap
            && weight.len() >= groups * n * k
            && weight_scale.len() >= groups * (n / 128) * (k / 128)
            && masked_m.len() >= groups
            && output.len() >= groups * cap * n,
        "GLM5.2 DeepGEMM masked grouped FP8 {kind:?} buffers too small: act {}, act_scale {}, w {}, w_scale {}, masked_m {}, out {}",
        activation.len(),
        activation_scale.len(),
        weight.len(),
        weight_scale.len(),
        masked_m.len(),
        output.len()
    );
    let (act_ptr, _act_guard) = activation.device_ptr(&ctx.stream);
    let (act_scale_ptr, _act_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (w_ptr, _w_guard) = weight.device_ptr(&ctx.stream);
    let (w_scale_ptr, _w_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_masked_grouped_fp8_launch_cuda(
            kind.abi(),
            act_ptr as *const u8,
            act_scale_ptr as *const f32,
            w_ptr as *const u8,
            w_scale_ptr as *const f32,
            masked_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 DeepGEMM masked grouped FP8 {kind:?} launch failed: {err}"))
}

/// Masked GEMM output `[32, 64, n]` → the aligned recv slots
/// `decode_combine` addresses (rows `offsets[g] + r` for `r < masked_m[g]`).
pub fn glm52_deepgemm_masked_out_to_aligned_launch(
    ctx: &DeviceContext,
    n: usize,
    masked_out: &CudaSlice<bf16>,
    masked_m: &CudaSlice<i32>,
    expert_offsets: &CudaSlice<i64>,
    aligned_out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let groups = GLM52_DEEPGEMM_MASKED_GROUPS;
    let cap = GLM52_DEEPGEMM_MASKED_CAP;
    ensure!(
        n > 0 && n.is_multiple_of(4),
        "GLM5.2 masked-out remap needs n % 4 == 0, got {n}"
    );
    ensure!(
        masked_out.len() >= groups * cap * n
            && masked_m.len() >= groups
            && expert_offsets.len() > groups,
        "GLM5.2 masked-out remap buffers too small: masked {}, masked_m {}, offsets {}",
        masked_out.len(),
        masked_m.len(),
        expert_offsets.len()
    );
    let (src_ptr, _src_guard) = masked_out.device_ptr(&ctx.stream);
    let (masked_ptr, _masked_guard) = masked_m.device_ptr(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = expert_offsets.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = aligned_out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_deepgemm_masked_out_to_aligned_cuda(
            src_ptr as *const ffi::Half,
            masked_ptr as *const i32,
            offsets_ptr as *const i64,
            dst_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 masked-out remap launch failed: {err}"))
}
