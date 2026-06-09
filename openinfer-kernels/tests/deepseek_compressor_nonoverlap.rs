#![cfg(feature = "deepseek-v4")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::CUstream;
use half::bf16;
use openinfer_kernels::ffi;

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

const ACTIVE_INPUT_DIMS: usize = 4;

fn active_input_term(token: usize, lane: usize, hidden_dim: usize) -> (usize, u16) {
    let k = (token * 97 + lane * 577 + 13) % hidden_dim;
    let token_component = (token % 31) as f32 - 15.0;
    let value = 0.015 * (lane + 1) as f32 + 0.0007 * token_component;
    (k, bf16_bits(value))
}

#[derive(Clone)]
struct NonOverlapCase {
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    ratio: usize,
    x: Vec<u16>,
    wkv: Vec<u16>,
    wgate: Vec<u16>,
    ape: Vec<f32>,
    norm: Vec<u16>,
}

fn make_case(seq_len: usize, hidden_dim: usize, head_dim: usize, ratio: usize) -> NonOverlapCase {
    let mut x = vec![bf16_bits(0.0); seq_len * hidden_dim];
    for token in 0..seq_len {
        for lane in 0..ACTIVE_INPUT_DIMS {
            let (k, value) = active_input_term(token, lane, hidden_dim);
            x[token * hidden_dim + k] = value;
        }
    }
    let mut wkv = vec![0u16; head_dim * hidden_dim];
    let mut wgate = vec![0u16; head_dim * hidden_dim];
    for out_dim in 0..head_dim {
        for k in 0..hidden_dim {
            let kv = ((out_dim % 17) as f32 - 8.0) * 0.001 + (k % 7) as f32 * 0.00003;
            let gate = ((out_dim % 13) as f32 - 6.0) * 0.0007 + (k % 5) as f32 * 0.00002;
            wkv[out_dim * hidden_dim + k] = bf16_bits(kv);
            wgate[out_dim * hidden_dim + k] = bf16_bits(gate);
        }
    }
    let mut ape = vec![0.0f32; ratio * head_dim];
    for route in 0..ratio {
        for dim in 0..head_dim {
            ape[route * head_dim + dim] =
                (route as f32 - ratio as f32 * 0.5) * 0.0001 + (dim % 19) as f32 * 0.00001;
        }
    }
    let norm = (0..head_dim)
        .map(|dim| bf16_bits(0.75 + (dim % 11) as f32 * 0.01))
        .collect();
    NonOverlapCase {
        seq_len,
        hidden_dim,
        head_dim,
        ratio,
        x,
        wkv,
        wgate,
        ape,
        norm,
    }
}

fn reference_nonoverlap(case: &NonOverlapCase, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let compressed_len = case.seq_len / case.ratio;
    let mut weighted = vec![0.0f32; compressed_len * case.head_dim];
    for compressed in 0..compressed_len {
        for dim in 0..case.head_dim {
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = [0.0f32; 128];
            let mut values = [0.0f32; 128];
            for route in 0..case.ratio {
                let token = compressed * case.ratio + route;
                let mut score = case.ape[route * case.head_dim + dim];
                let mut value = 0.0f32;
                for lane in 0..ACTIVE_INPUT_DIMS {
                    let (k, _) = active_input_term(token, lane, case.hidden_dim);
                    let x = bf16_f32(case.x[token * case.hidden_dim + k]);
                    score += x * bf16_f32(case.wgate[dim * case.hidden_dim + k]);
                    value += x * bf16_f32(case.wkv[dim * case.hidden_dim + k]);
                }
                scores[route] = score;
                values[route] = value;
                max_score = max_score.max(score);
            }

            let mut denom = 0.0f32;
            let mut acc = 0.0f32;
            for route in 0..case.ratio {
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

fn run_nonoverlap(case: &NonOverlapCase, eps: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    let compressed_len = case.seq_len / case.ratio;
    let x_d = DeviceBuffer::from_host(&case.x)?;
    let wkv_d = DeviceBuffer::from_host(&case.wkv)?;
    let wgate_d = DeviceBuffer::from_host(&case.wgate)?;
    let ape_d = DeviceBuffer::from_host(&case.ape)?;
    let norm_d = DeviceBuffer::from_host(&case.norm)?;
    let weighted_d = DeviceBuffer::<f32>::zeroed(compressed_len * case.head_dim)?;
    let out_d = DeviceBuffer::<u16>::zeroed(compressed_len * case.head_dim)?;
    let stream: CUstream = ptr::null_mut();
    let result = unsafe {
        ffi::deepseek_compressor_nonoverlap_prefill_cuda(
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
            case.ratio as i32,
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

#[derive(Debug)]
struct DiffStats {
    max_abs: f32,
    first_diff: Option<(usize, f32, f32)>,
}

fn diff_stats(got: &[f32], expected: &[f32]) -> Result<DiffStats> {
    ensure!(got.len() == expected.len(), "length mismatch");
    let mut max_abs = 0.0f32;
    let mut first_diff = None;
    for (idx, (&a, &b)) in got.iter().zip(expected).enumerate() {
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        if first_diff.is_none() && abs > 0.0 {
            first_diff = Some((idx, a, b));
        }
    }
    Ok(DiffStats {
        max_abs,
        first_diff,
    })
}

fn check_case(
    name: &str,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    ratio: usize,
) -> Result<()> {
    let eps = 1.0e-6;
    let case = make_case(seq_len, hidden_dim, head_dim, ratio);
    let (expected_weighted, expected_out) = reference_nonoverlap(&case, eps);
    let (got_weighted, got_out) = run_nonoverlap(&case, eps)
        .with_context(|| format!("running non-overlap compressor case {name}"))?;
    let weighted_stats = diff_stats(&got_weighted, &expected_weighted)?;
    let out_stats = diff_stats(&got_out, &expected_out)?;
    println!(
        "{name}: weighted max_abs={} first_diff={:?}; out max_abs={} first_diff={:?}",
        weighted_stats.max_abs, weighted_stats.first_diff, out_stats.max_abs, out_stats.first_diff
    );
    ensure!(
        weighted_stats.max_abs <= 5.0e-3,
        "{name} weighted max_abs {} > 5e-3",
        weighted_stats.max_abs
    );
    ensure!(
        out_stats.max_abs <= 5.0e-3,
        "{name} out max_abs {} > 5e-3",
        out_stats.max_abs
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; locks #145 non-overlap compressor real-shape acceptance"]
fn nonoverlap_prefill_matches_real_shape_contract() -> Result<()> {
    // Shape source: PR #145 5090 acceptance plus DSV4 `compress_ratios`, which
    // exercises ratio4 and ratio128 layers. These are correctness gates for
    // legal runtime shapes, not standalone performance claims.
    check_case("ratio2-odd", 21, 64, 32, 2)?;
    check_case("ratio4-representative", 20, 64, 32, 4)?;
    check_case("ratio128-minimal", 128, 64, 32, 128)?;
    check_case("10k-ratio2", 10580, 4096, 512, 2)?;
    check_case("10k-ratio128", 10580, 4096, 512, 128)?;
    Ok(())
}
