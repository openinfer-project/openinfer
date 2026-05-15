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
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, stop: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
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

// New CuTeDSL Sm120 GEMM path constrains:
//   - hidden_dim == 4096 (config.dim baked into the AOT cubin)
//   - head_dim ∈ {128 (indexer), 512 (main)}
//   - seq_len divisible by ratio=4; M is padded internally to next 128 multiple.
// Tolerances are looser than the retired scalar serial kernel because BF16
// tensor-core MMA accumulates in a different order than per-k scalar FMA;
// for the all-ones x test data this still keeps relative error well below
// downstream BF16 quantisation noise.
fn check_case(name: &str, seq_len: usize, head_dim: usize) -> Result<()> {
    ensure!(seq_len % 4 == 0, "seq_len must be ratio4 aligned");
    let hidden_dim = 4096;
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
#[ignore = "requires CUDA GPU; covers 10k launch shape for main and indexer calls"]
fn overlap_prefill_matches_reference_10k_representative_shapes() -> Result<()> {
    check_case("10k-indexer", 10580, 128)?;
    check_case("10k-main", 10580, 512)?;
    Ok(())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() as f32 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_overlap_prefill(
    name: &str,
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    warmup: usize,
    iters: usize,
) -> Result<()> {
    let eps = 1.0e-6;
    let case = make_case(seq_len, hidden_dim, head_dim);
    let compressed_len = seq_len / 4;

    let x_d = DeviceBuffer::from_host(&case.x)?;
    let wkv_d = DeviceBuffer::from_host(&case.wkv)?;
    let wgate_d = DeviceBuffer::from_host(&case.wgate)?;
    let ape_d = DeviceBuffer::from_host(&case.ape)?;
    let norm_d = DeviceBuffer::from_host(&case.norm)?;
    let weighted_d = DeviceBuffer::<f32>::zeroed(compressed_len * head_dim)?;
    let out_d = DeviceBuffer::<u16>::zeroed(compressed_len * head_dim)?;
    let stream: CUstream = ptr::null_mut();

    let launch = || -> Result<()> {
        let result = unsafe {
            ffi::deepseek_compressor_overlap_prefill_cuda(
                x_d.ptr,
                wkv_d.ptr,
                wgate_d.ptr,
                ape_d.ptr,
                norm_d.ptr,
                weighted_d.ptr,
                out_d.ptr,
                seq_len as i32,
                hidden_dim as i32,
                head_dim as i32,
                eps,
                stream,
            )
        };
        ensure!(
            result == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{name} launch failed: {:?}",
            result
        );
        Ok(())
    };

    for _ in 0..warmup {
        launch()?;
    }
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let mut start_evt: *mut c_void = ptr::null_mut();
    let mut stop_evt: *mut c_void = ptr::null_mut();
    cuda_check(unsafe { cudaEventCreate(&mut start_evt) })?;
    cuda_check(unsafe { cudaEventCreate(&mut stop_evt) })?;

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        cuda_check(unsafe { cudaEventRecord(start_evt, ptr::null_mut()) })?;
        launch()?;
        cuda_check(unsafe { cudaEventRecord(stop_evt, ptr::null_mut()) })?;
        cuda_check(unsafe { cudaEventSynchronize(stop_evt) })?;
        let mut ms: f32 = 0.0;
        cuda_check(unsafe { cudaEventElapsedTime(&mut ms, start_evt, stop_evt) })?;
        samples.push(ms);
    }

    cuda_check(unsafe { cudaEventDestroy(start_evt) })?;
    cuda_check(unsafe { cudaEventDestroy(stop_evt) })?;

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f32 = samples.iter().sum();
    let avg = sum / samples.len() as f32;
    let p50 = percentile(&samples, 0.50);
    let p95 = percentile(&samples, 0.95);
    let p99 = percentile(&samples, 0.99);
    let min = samples[0];
    let max = *samples.last().unwrap();

    // Equivalent throughput: how many compressed positions emitted per second.
    let tok_per_s = compressed_len as f32 / (avg / 1000.0);

    println!(
        "[microbench] {name} seq_len={seq_len} hidden_dim={hidden_dim} head_dim={head_dim} \
         compressed_len={compressed_len} warmup={warmup} iters={iters}\n\
         \tms/call: avg={avg:.3} p50={p50:.3} p95={p95:.3} p99={p99:.3} min={min:.3} max={max:.3}\n\
         \tcompressed_pos/s={tok_per_s:.1}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; microbench overlap compressor prefill at production hidden_dim"]
fn overlap_prefill_microbench_production_shapes() -> Result<()> {
    // DSV4 production: hidden_dim = config.dim = 4096.
    // head_dim 128 corresponds to the indexer compressor call site,
    // head_dim 512 corresponds to the main compressor call site.
    let seq_len = env_usize("OVERLAP_BENCH_SEQ_LEN", 10580);
    let hidden_dim = env_usize("OVERLAP_BENCH_HIDDEN_DIM", 4096);
    let warmup = env_usize("OVERLAP_BENCH_WARMUP", 10);
    let iters = env_usize("OVERLAP_BENCH_ITERS", 50);

    bench_overlap_prefill("10k-indexer", seq_len, hidden_dim, 128, warmup, iters)?;
    bench_overlap_prefill("10k-main", seq_len, hidden_dim, 512, warmup, iters)?;
    Ok(())
}
