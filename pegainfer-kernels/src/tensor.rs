//! Device tensor types and CUDA context.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::ffi;

/// Marker trait for tensor metadata tags.
pub trait NamedTag {
    const NAME: &'static str;
}

/// Marker trait for tensor element type vocabulary.
pub trait DTypeTag: NamedTag {}

/// Marker trait for tensor layout vocabulary.
pub trait LayoutTag: NamedTag {}

/// Marker trait for tensor axis vocabulary.
pub trait AxisTag: NamedTag {}

macro_rules! named_tag {
    ($name:ident, $value:literal, $trait_name:ident) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl NamedTag for $name {
            const NAME: &'static str = $value;
        }

        impl $trait_name for $name {}
    };
}

named_tag!(Bf16, "bf16", DTypeTag);
named_tag!(F32, "f32", DTypeTag);
named_tag!(I32, "i32", DTypeTag);
named_tag!(U32, "u32", DTypeTag);
named_tag!(U8, "u8", DTypeTag);

named_tag!(Contiguous1D, "contiguous_1d", LayoutTag);
named_tag!(RowMajor2D, "row_major_2d", LayoutTag);
named_tag!(HiddenStatesLayout, "hidden_states", LayoutTag);
named_tag!(PagedKvPageFirst, "paged_kv_page_first", LayoutTag);

named_tag!(Batch, "batch", AxisTag);
named_tag!(BatchPlusOne, "batch_plus_1", AxisTag);
named_tag!(HeadDim, "head_dim", AxisTag);
named_tag!(Hidden, "hidden", AxisTag);
named_tag!(InDim, "in", AxisTag);
named_tag!(Intermediate, "intermediate", AxisTag);
named_tag!(Inter2, "inter2", AxisTag);
named_tag!(Kv, "kv", AxisTag);
named_tag!(KvDim, "kv_dim", AxisTag);
named_tag!(KvHead, "kv_head", AxisTag);
named_tag!(Layer, "layer", AxisTag);
named_tag!(OutDim, "out", AxisTag);
named_tag!(OutTotal, "out_total", AxisTag);
named_tag!(Page, "page", AxisTag);
named_tag!(PageSlot, "page_slot", AxisTag);
named_tag!(PosInPage, "pos_in_page", AxisTag);
named_tag!(QDim, "q_dim", AxisTag);
named_tag!(RopeDim, "rope_dim", AxisTag);
named_tag!(Seq, "seq", AxisTag);
named_tag!(Tile, "tile", AxisTag);
named_tag!(Token, "token", AxisTag);
named_tag!(Vocab, "vocab", AxisTag);

/// One named axis in an erased tensor metadata description.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct AxisSpec {
    pub name: String,
    pub size: usize,
}

impl AxisSpec {
    pub fn new<A: AxisTag>(size: usize) -> Self {
        Self {
            name: A::NAME.to_string(),
            size,
        }
    }

    pub fn named(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

/// Erased tensor metadata for schedules, reports, and future instrumentation.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TensorSpec {
    pub dtype: String,
    pub layout: String,
    pub axes: Vec<AxisSpec>,
}

impl TensorSpec {
    pub fn new<D: DTypeTag, L: LayoutTag>(axes: impl IntoIterator<Item = AxisSpec>) -> Self {
        Self {
            dtype: D::NAME.to_string(),
            layout: L::NAME.to_string(),
            axes: axes.into_iter().collect(),
        }
    }

    pub fn named(
        dtype: impl Into<String>,
        layout: impl Into<String>,
        axes: impl IntoIterator<Item = AxisSpec>,
    ) -> Self {
        Self {
            dtype: dtype.into(),
            layout: layout.into(),
            axes: axes.into_iter().collect(),
        }
    }

    pub fn compact(&self) -> String {
        let axes = self
            .axes
            .iter()
            .map(|axis| format!("{}={}", axis.name, axis.size))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}[{}] layout={}", self.dtype, axes, self.layout)
    }
}

/// A named kernel argument carrying an erased tensor description.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TensorArg {
    pub name: String,
    pub spec: TensorSpec,
}

impl TensorArg {
    pub fn new(name: impl Into<String>, spec: TensorSpec) -> Self {
        Self {
            name: name.into(),
            spec,
        }
    }

    pub fn compact(&self) -> String {
        format!("{}: {}", self.name, self.spec.compact())
    }
}

/// String-valued non-tensor kernel metadata.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct AttrSpec {
    pub name: String,
    pub value: String,
}

impl AttrSpec {
    pub fn new(name: impl Into<String>, value: String) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// Erased logical kernel call IR shared by static schedules and future traces.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct KernelCall {
    pub op: String,
    pub label: String,
    pub inputs: Vec<TensorArg>,
    pub outputs: Vec<TensorArg>,
    pub attrs: Vec<AttrSpec>,
}

impl KernelCall {
    #[must_use]
    pub fn new(op: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            label: label.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: Vec::new(),
        }
    }

    #[must_use]
    pub fn input(mut self, name: impl Into<String>, spec: TensorSpec) -> Self {
        self.inputs.push(TensorArg::new(name, spec));
        self
    }

    #[must_use]
    pub fn output(mut self, name: impl Into<String>, spec: TensorSpec) -> Self {
        self.outputs.push(TensorArg::new(name, spec));
        self
    }

    #[must_use]
    pub fn attr(mut self, name: impl Into<String>, value: String) -> Self {
        self.attrs.push(AttrSpec::new(name, value));
        self
    }
}

/// CUDA device context holding context and stream.
#[derive(Clone)]
pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub device_ordinal: usize,
}

impl DeviceContext {
    pub fn new() -> Result<Self> {
        Self::new_with_device(0)
    }

    pub fn new_with_device(device_ordinal: usize) -> Result<Self> {
        unsafe {
            let err = ffi::cuda_set_device(device_ordinal as i32);
            if err != 0 {
                return Err(anyhow!(
                    "Failed to set CUDA device {}: cudaError={}",
                    device_ordinal,
                    err
                ));
            }
        }
        let ctx = CudaContext::new(device_ordinal)
            .map_err(|e| anyhow!("Failed to create CUDA context: {}", e))?;

        // Disable multi-stream event tracking before creating streams.
        // We use a single compute stream, so no cross-stream synchronization is needed.
        // This avoids stream.wait(event) calls that break CUDA Graph capture.
        // SAFETY: We only use one stream for all GPU work.
        unsafe {
            ctx.disable_event_tracking();
        }

        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA stream: {}", e))?;

        // Initialize cuBLAS handle
        unsafe {
            ffi::cublas_init();
        }

        Ok(Self {
            ctx,
            stream,
            device_ordinal,
        })
    }

    /// Synchronize stream
    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("Sync failed: {}", e))
    }
}

/// 1D device tensor (vector) — stored as bf16.
pub struct DeviceVec {
    pub data: CudaSlice<bf16>,
    pub len: usize,
}

impl DeviceVec {
    /// Create from host data (bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16]) -> Result<Self> {
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len: data.len(),
        })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_safetensors(ctx: &DeviceContext, data: &[u8]) -> Result<Self> {
        if !data.len().is_multiple_of(2) {
            return Err(anyhow!(
                "Data length must be even for bf16: got {} bytes",
                data.len()
            ));
        }
        let len = data.len() / 2;
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), len) };
        Self::from_host(ctx, slice)
    }

    /// Create zeroed tensor
    pub fn zeros(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let gpu_data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len,
        })
    }

    /// Copy to host as f32.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }
}

impl Clone for DeviceVec {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            len: self.len,
        }
    }
}

/// 2D device tensor (matrix) — stored in row-major order as bf16.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceMatrix {
    /// Vertically stack matrices (same cols, concatenate rows). GPU D2D copy.
    pub fn vstack(ctx: &DeviceContext, matrices: &[&DeviceMatrix]) -> Result<Self> {
        assert!(!matrices.is_empty());
        let cols = matrices[0].cols;
        for m in matrices {
            assert_eq!(m.cols, cols, "vstack: all matrices must have same cols");
        }
        let total_rows: usize = matrices.iter().map(|m| m.rows).sum();
        let mut data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_rows * cols)
            .map_err(|e| anyhow!("vstack alloc failed: {}", e))?;
        let mut offset = 0;
        for m in matrices {
            let n = m.rows * m.cols;
            let src = m.data.slice(..n);
            let mut dst = data.slice_mut(offset..offset + n);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("vstack D2D copy failed: {}", e))?;
            offset += n;
        }
        Ok(Self {
            data,
            rows: total_rows,
            cols,
        })
    }

    /// Create from host data (row-major, bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
        })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_safetensors(
        ctx: &DeviceContext,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols * std::mem::size_of::<bf16>() {
            return Err(anyhow!(
                "Data length mismatch: expected {} bytes, got {} bytes",
                rows * cols * std::mem::size_of::<bf16>(),
                data.len()
            ));
        }
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * cols) };
        Self::from_host(ctx, slice, rows, cols)
    }
}

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
/// Memory layout: [hidden_dim * seq_len] elements, token i at offset i * hidden_dim.
/// cuBLAS interprets as [hidden_dim, seq_len] column-major.
pub struct HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl HiddenStates {
    /// Create zeroed batch
    pub fn zeros(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(hidden_dim * seq_len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy_matrix_to_host(ctx: &DeviceContext, matrix: &DeviceMatrix) -> Vec<bf16> {
        let host = ctx
            .stream
            .clone_dtoh(&matrix.data)
            .expect("D2H copy failed");
        ctx.sync().expect("CUDA sync failed");
        host
    }

    #[test]
    fn test_device_matrix_from_host_roundtrip() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 2;
        let cols = 3;
        let host = vec![
            bf16::from_f32(-1.5),
            bf16::from_f32(0.0),
            bf16::from_f32(2.25),
            bf16::from_f32(7.0),
            bf16::from_f32(-3.0),
            bf16::from_f32(0.5),
        ];

        let matrix =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);

        let got = copy_matrix_to_host(&ctx, &matrix);
        assert_eq!(got.len(), host.len());
        for (idx, (actual, expected)) in got.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip mismatch at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_device_matrix_from_safetensors_matches_from_host() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 3;
        let cols = 2;
        let host = vec![
            bf16::from_f32(-8.0),
            bf16::from_f32(-0.25),
            bf16::from_f32(1.0),
            bf16::from_f32(3.5),
            bf16::from_f32(9.0),
            bf16::from_f32(10.75),
        ];
        let safetensor_bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let from_host =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");
        let from_safetensors = DeviceMatrix::from_safetensors(&ctx, &safetensor_bytes, rows, cols)
            .expect("from_safetensors should succeed");

        assert_eq!(from_safetensors.rows, from_host.rows);
        assert_eq!(from_safetensors.cols, from_host.cols);

        let host_out = copy_matrix_to_host(&ctx, &from_host);
        let safetensors_out = copy_matrix_to_host(&ctx, &from_safetensors);
        assert_eq!(host_out.len(), safetensors_out.len());
        for (idx, (a, b)) in host_out.iter().zip(safetensors_out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "from_safetensors/from_host mismatch at index {}",
                idx
            );
        }
    }
}
