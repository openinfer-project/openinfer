#![cfg(feature = "deepseek-v4")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::CUstream;
use half::bf16;
use pegainfer_kernels::ffi;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy + Default> DeviceBuffer<T> {
    fn from_host(data: &[T]) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let bytes = std::mem::size_of_val(data);
        cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) })?;
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    ptr,
                    data.as_ptr().cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            })?;
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len: data.len(),
        })
    }

    fn zeroed(len: usize) -> Result<Self> {
        Self::from_host(&vec![T::default(); len])
    }

    fn copy_to_host(&self) -> Result<Vec<T>> {
        let mut data = vec![T::default(); self.len];
        let bytes = self.len * size_of::<T>();
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    data.as_mut_ptr().cast::<c_void>(),
                    self.ptr.cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
        }
        Ok(data)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr.cast::<c_void>());
            }
        }
    }
}

fn cuda_check(code: i32) -> Result<()> {
    ensure!(code == 0, "CUDA runtime call failed with code {code}");
    Ok(())
}

fn bf16_bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

fn bf16_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

#[derive(Clone)]
struct OverlapCase {
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    x: Vec<u16>,
    wkv: Vec<u16>,
    wgate: Vec<u16>,
    ape: Vec<f32>,
    norm: Vec<u16>,
}

fn make_case(seq_len: usize, hidden_dim: usize, head_dim: usize) -> OverlapCase {
    let x = vec![bf16_bits(1.0); seq_len * hidden_dim];
    let mut wkv = vec![0u16; 2 * head_dim * hidden_dim];
    let mut wgate = vec![0u16; 2 * head_dim * hidden_dim];
    for out_dim in 0..2 * head_dim {
        for k in 0..hidden_dim {
            let kv = ((out_dim % 17) as f32 - 8.0) * 0.001 + (k % 7) as f32 * 0.00003;
            let gate = ((out_dim % 13) as f32 - 6.0) * 0.0007 + (k % 5) as f32 * 0.00002;
            wkv[out_dim * hidden_dim + k] = bf16_bits(kv);
            wgate[out_dim * hidden_dim + k] = bf16_bits(gate);
        }
    }
    let mut ape = vec![0.0f32; 4 * 2 * head_dim];
    for route in 0..4 {
        for dim in 0..2 * head_dim {
            ape[route * 2 * head_dim + dim] =
                (route as f32 - 1.5) * 0.01 + (dim % 19) as f32 * 0.0001;
        }
    }
    let norm = (0..head_dim)
        .map(|dim| bf16_bits(0.75 + (dim % 11) as f32 * 0.01))
        .collect();
    OverlapCase {
        seq_len,
        hidden_dim,
        head_dim,
        x,
        wkv,
        wgate,
        ape,
        norm,
    }
}

fn reference_overlap(case: &OverlapCase, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let compressed_len = case.seq_len / 4;
    let routes = 8;
    let mut wkv_sums = vec![0.0f32; 2 * case.head_dim];
    let mut wgate_sums = vec![0.0f32; 2 * case.head_dim];
    for out_dim in 0..2 * case.head_dim {
        for k in 0..case.hidden_dim {
            wkv_sums[out_dim] += bf16_f32(case.wkv[out_dim * case.hidden_dim + k]);
            wgate_sums[out_dim] += bf16_f32(case.wgate[out_dim * case.hidden_dim + k]);
        }
    }

    let mut weighted = vec![0.0f32; compressed_len * case.head_dim];
    for compressed in 0..compressed_len {
        for dim in 0..case.head_dim {
            let mut scores = [0.0f32; 8];
            let mut values = [0.0f32; 8];
            for route in 0..routes {
                let (valid, out_dim, ape_dim) = if route < 4 {
                    (compressed > 0, dim, route * (2 * case.head_dim) + dim)
                } else {
                    let local_route = route - 4;
                    (
                        true,
                        case.head_dim + dim,
                        local_route * (2 * case.head_dim) + case.head_dim + dim,
                    )
                };
                if valid {
                    scores[route] = wgate_sums[out_dim] + case.ape[ape_dim];
                    values[route] = wkv_sums[out_dim];
                } else {
                    scores[route] = -3.4028234663852886e38f32;
                    values[route] = 0.0;
                }
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            let mut acc = 0.0f32;
            for route in 0..routes {
                let prob = (scores[route] - max_score).exp();
                denom += prob;
                acc += prob * values[route];
            }
            weighted[compressed * case.head_dim + dim] = acc / denom;
        }
    }

    let mut out = vec![0.0f32; compressed_len * case.head_dim];
    for compressed in 0..compressed_len {
        let mut sum_sq = 0.0f32;
        for dim in 0..case.head_dim {
            let value = weighted[compressed * case.head_dim + dim];
            sum_sq += value * value;
        }
        let inv_rms = (sum_sq / case.head_dim as f32 + eps).sqrt().recip();
        for dim in 0..case.head_dim {
            let value =
                weighted[compressed * case.head_dim + dim] * inv_rms * bf16_f32(case.norm[dim]);
            out[compressed * case.head_dim + dim] = bf16::from_f32(value).to_f32();
        }
    }
    (weighted, out)
}

fn run_overlap(case: &OverlapCase, eps: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    let compressed_len = case.seq_len / 4;
    let x_d = DeviceBuffer::from_host(&case.x)?;
    let wkv_d = DeviceBuffer::from_host(&case.wkv)?;
    let wgate_d = DeviceBuffer::from_host(&case.wgate)?;
    let ape_d = DeviceBuffer::from_host(&case.ape)?;
    let norm_d = DeviceBuffer::from_host(&case.norm)?;
    let weighted_d = DeviceBuffer::<f32>::zeroed(compressed_len * case.head_dim)?;
    let out_d = DeviceBuffer::<u16>::zeroed(compressed_len * case.head_dim)?;
    let stream: CUstream = ptr::null_mut();
    let result = unsafe {
        ffi::deepseek_compressor_overlap_prefill_cuda(
            x_d.ptr,
            wkv_d.ptr,
            wgate_d.ptr,
            ape_d.ptr,
            norm_d.ptr,
            weighted_d.ptr,
            out_d.ptr,
            case.seq_len as i32,
            case.hidden_dim as i32,
            case.head_dim as i32,
            eps,
            stream,
        )
    };
    assert_eq!(result, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;
    let weighted = weighted_d.copy_to_host()?;
    let out = out_d.copy_to_host()?.into_iter().map(bf16_f32).collect();
    Ok((weighted, out))
}

fn assert_close(name: &str, got: &[f32], expected: &[f32], max_abs_limit: f32) -> Result<()> {
    ensure!(got.len() == expected.len(), "{name} length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (idx, (&a, &b)) in got.iter().zip(expected).enumerate() {
        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
            max_idx = idx;
        }
    }
    ensure!(
        max_abs <= max_abs_limit,
        "{name} max_abs {max_abs} > {max_abs_limit} at {max_idx}: got={} expected={}",
        got[max_idx],
        expected[max_idx]
    );
    Ok(())
}

// Tolerances are looser than what a scalar BF16 FMA kernel produces because
// the new path routes both dot products through cuBLAS BF16 tensor-core MMA,
// which accumulates in a different order than the per-k scalar FMA the
// reference uses; the 3x bf16-noise FlashAttention-style envelope is well
// inside what the downstream RMSNorm BF16 store quantises away anyway.
fn check_case(name: &str, seq_len: usize, hidden_dim: usize, head_dim: usize) -> Result<()> {
    ensure!(seq_len % 4 == 0, "seq_len must be ratio4 aligned");
    let eps = 1.0e-6;
    let case = make_case(seq_len, hidden_dim, head_dim);
    let (expected_weighted, expected_out) = reference_overlap(&case, eps);
    let (got_weighted, got_out) = run_overlap(&case, eps)
        .with_context(|| format!("running overlap compressor case {name}"))?;
    assert_close(
        &format!("{name} weighted"),
        &got_weighted,
        &expected_weighted,
        5.0e-3,
    )?;
    assert_close(&format!("{name} out"), &got_out, &expected_out, 5.0e-3)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; validates overlap compressor prefill core"]
fn overlap_prefill_matches_reference_small_main_and_indexer_shapes() -> Result<()> {
    check_case("main-small", 20, 64, 32)?;
    check_case("indexer-small", 20, 64, 16)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; covers odd/boundary compressed and head shapes"]
fn overlap_prefill_matches_reference_odd_boundary_shape() -> Result<()> {
    check_case("odd-boundary", 68, 40, 33)
}

#[test]
#[ignore = "requires CUDA GPU; covers 10k launch shape for main and indexer calls"]
fn overlap_prefill_matches_reference_10k_representative_shapes() -> Result<()> {
    check_case("10k-indexer", 10580, 4096, 128)?;
    check_case("10k-main", 10580, 4096, 512)?;
    Ok(())
}
