#![cfg(feature = "deepseek-v4-cutedsl-diagnostic")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::{Result, ensure};
use cudarc::driver::sys::{CUresult, CUstream};
use half::bf16;
use pegainfer_kernels::ffi;

const HEAD_DIM: usize = 128;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;

    fn deepseek_cutedsl_indexer_dots_bf16_cuda(
        q: *const ffi::Half,
        kv: *const ffi::Half,
        dots: *mut f32,
        rows: i32,
        compressed_len: i32,
        stream: CUstream,
    ) -> CUresult;

    fn deepseek_cutedsl_indexer_scores_exact_bf16_cuda(
        q: *const ffi::Half,
        kv: *const ffi::Half,
        weights: *const ffi::Half,
        scores: *mut f32,
        seq_len: i32,
        compressed_len: i32,
        score_scale: f32,
        stream: CUstream,
    ) -> CUresult;
}

struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy + Default> DeviceBuffer<T> {
    fn from_host(data: &[T]) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let bytes = data.len() * size_of::<T>();
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

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
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

fn assert_cuda_success(result: CUresult) {
    assert_eq!(result, CUresult::CUDA_SUCCESS);
}

fn patterned_bf16(len: usize, scale: f32, bias: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let v = ((i * 37 + 11) % 29) as f32 - 14.0;
            bf16::from_f32(v * scale + bias)
        })
        .collect()
}

fn lcg_bf16(len: usize, seed: u64, scale: f32, bias: f32) -> Vec<bf16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 40) & 0xffff) as u32;
            let centered = bits as f32 / 32768.0 - 1.0;
            bf16::from_f32(centered * scale + bias)
        })
        .collect()
}

fn bf16_words(values: &[bf16]) -> Vec<ffi::Half> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reference_prefill(
    q: &[bf16],
    kv: &[bf16],
    weights: &[bf16],
    seq_len: usize,
    local_heads: usize,
    compressed_len: usize,
    score_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; seq_len * compressed_len];
    for token in 0..seq_len {
        for compressed in 0..compressed_len {
            let mut acc = 0.0f32;
            for head in 0..local_heads {
                let row = token * local_heads + head;
                let mut dot = 0.0f32;
                for k in 0..HEAD_DIM {
                    dot += q[row * HEAD_DIM + k].to_f32() * kv[compressed * HEAD_DIM + k].to_f32();
                }
                acc += dot.max(0.0) * weights[row].to_f32();
            }
            scores[token * compressed_len + compressed] = acc * score_scale;
        }
    }
    scores
}

fn reference_decode(
    q: &[bf16],
    kv: &[bf16],
    weights: &[bf16],
    local_heads: usize,
    compressed_len: usize,
    score_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; compressed_len];
    for compressed in 0..compressed_len {
        let mut acc = 0.0f32;
        for head in 0..local_heads {
            let mut dot = 0.0f32;
            for k in 0..HEAD_DIM {
                dot += q[head * HEAD_DIM + k].to_f32() * kv[compressed * HEAD_DIM + k].to_f32();
            }
            acc += dot.max(0.0) * weights[head].to_f32();
        }
        scores[compressed] = acc * score_scale;
    }
    scores
}

fn cutedsl_scores(
    q_d: &DeviceBuffer<ffi::Half>,
    kv_d: &DeviceBuffer<ffi::Half>,
    weights: &[bf16],
    seq_len: usize,
    local_heads: usize,
    compressed_len: usize,
    score_scale: f32,
    stream: CUstream,
) -> Result<Vec<f32>> {
    let rows = seq_len * local_heads;
    let mut dots_d = DeviceBuffer::from_host(&vec![0.0f32; rows * compressed_len])?;
    let result = unsafe {
        deepseek_cutedsl_indexer_dots_bf16_cuda(
            q_d.as_ptr(),
            kv_d.as_ptr(),
            dots_d.as_mut_ptr(),
            rows as i32,
            compressed_len as i32,
            stream,
        )
    };
    assert_cuda_success(result);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let dots = dots_d.copy_to_host()?;
    let mut scores = vec![0.0f32; seq_len * compressed_len];
    for token in 0..seq_len {
        for compressed in 0..compressed_len {
            let mut acc = 0.0f32;
            for head in 0..local_heads {
                let row = token * local_heads + head;
                acc += dots[row * compressed_len + compressed].max(0.0) * weights[row].to_f32();
            }
            scores[token * compressed_len + compressed] = acc * score_scale;
        }
    }
    Ok(scores)
}

fn cutedsl_exact_scores(
    q_d: &DeviceBuffer<ffi::Half>,
    kv_d: &DeviceBuffer<ffi::Half>,
    weights_d: &DeviceBuffer<ffi::Half>,
    seq_len: usize,
    compressed_len: usize,
    score_scale: f32,
    stream: CUstream,
) -> Result<Vec<f32>> {
    let mut scores_d = DeviceBuffer::from_host(&vec![0.0f32; seq_len * compressed_len])?;
    let result = unsafe {
        deepseek_cutedsl_indexer_scores_exact_bf16_cuda(
            q_d.as_ptr(),
            kv_d.as_ptr(),
            weights_d.as_ptr(),
            scores_d.as_mut_ptr(),
            seq_len as i32,
            compressed_len as i32,
            score_scale,
            stream,
        )
    };
    assert_cuda_success(result);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;
    scores_d.copy_to_host()
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let diff = (got - expected).abs();
        let tol = 2.5e-3f32.max(expected.abs() * 2.5e-2);
        assert!(
            diff <= tol,
            "mismatch at {idx}: got={got}, expected={expected}, diff={diff}, tol={tol}"
        );
    }
}

fn topk_indices(
    scores: &[f32],
    token: usize,
    compressed_len: usize,
    topk: usize,
    ratio: usize,
) -> Vec<i32> {
    let valid = ((token + 1) / ratio).min(compressed_len);
    let mut selected = vec![false; compressed_len];
    let mut out = Vec::with_capacity(topk);
    for _ in 0..topk {
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;
        for candidate in 0..valid {
            if selected[candidate] {
                continue;
            }
            let score = scores[token * compressed_len + candidate];
            if score > best_score {
                best_score = score;
                best_idx = Some(candidate);
            }
        }
        if let Some(idx) = best_idx {
            selected[idx] = true;
            out.push(idx as i32);
        } else {
            out.push(-1);
        }
    }
    out
}

#[test]
fn indexer_scores_prefill_cutedsl_aot_matches_reference() -> Result<()> {
    let seq_len = 4usize;
    let local_heads = 8usize;
    let compressed_len = 16usize;
    let score_scale = 0.125f32;
    let rows = seq_len * local_heads;

    let q = patterned_bf16(rows * HEAD_DIM, 0.003, 0.001);
    let kv = patterned_bf16(compressed_len * HEAD_DIM, -0.002, 0.002);
    let weights = patterned_bf16(rows, 0.01, 0.05);
    let expected = reference_prefill(
        &q,
        &kv,
        &weights,
        seq_len,
        local_heads,
        compressed_len,
        score_scale,
    );

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let stream: CUstream = ptr::null_mut();

    let got = cutedsl_scores(
        &q_d,
        &kv_d,
        &weights,
        seq_len,
        local_heads,
        compressed_len,
        score_scale,
        stream,
    )?;
    assert_close(&got, &expected);
    Ok(())
}

#[test]
fn indexer_scores_prefill_cutedsl_exact_matches_serial_topk_shape() -> Result<()> {
    let seq_len = 512usize;
    let local_heads = 8usize;
    let compressed_len = 129usize;
    let topk = 32usize;
    let ratio = 4usize;
    let score_scale = 0.125f32;
    let rows = seq_len * local_heads;

    let q = lcg_bf16(rows * HEAD_DIM, 0x3141_5926, 0.06, 0.0);
    let kv = lcg_bf16(compressed_len * HEAD_DIM, 0x2718_2818, 0.06, 0.0);
    let weights = lcg_bf16(rows, 0xfeed_beef, 0.08, 0.04);

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let weights_d = DeviceBuffer::from_host(&bf16_words(&weights))?;
    let mut serial_scores_d = DeviceBuffer::from_host(&vec![0.0f32; seq_len * compressed_len])?;
    let stream: CUstream = ptr::null_mut();

    let cutedsl = cutedsl_exact_scores(
        &q_d,
        &kv_d,
        &weights_d,
        seq_len,
        compressed_len,
        score_scale,
        stream,
    )?;
    let result = unsafe {
        ffi::deepseek_indexer_scores_prefill_cuda(
            q_d.as_ptr(),
            kv_d.as_ptr(),
            weights_d.as_ptr(),
            serial_scores_d.as_mut_ptr(),
            seq_len as i32,
            local_heads as i32,
            HEAD_DIM as i32,
            compressed_len as i32,
            score_scale,
            stream,
        )
    };
    assert_cuda_success(result);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let serial = serial_scores_d.copy_to_host()?;
    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    let mut max_rel = 0.0f32;
    for (idx, (&got, &expected)) in cutedsl.iter().zip(&serial).enumerate() {
        let abs = (got - expected).abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_idx = idx;
        }
        let rel = abs / expected.abs().max(1e-6);
        if rel > max_rel {
            max_rel = rel;
        }
    }

    let mut mismatches = Vec::new();
    for token in 0..seq_len {
        let cutedsl_topk = topk_indices(&cutedsl, token, compressed_len, topk, ratio);
        let serial_topk = topk_indices(&serial, token, compressed_len, topk, ratio);
        if cutedsl_topk != serial_topk {
            mismatches.push((token, cutedsl_topk, serial_topk));
            if mismatches.len() >= 5 {
                break;
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "CuTeDSL exact score path changed top-k: first mismatches={mismatches:?}, max_abs={max_abs} at {max_abs_idx}, max_rel={max_rel}"
    );
    assert!(
        max_abs == 0.0,
        "CuTeDSL exact score path changed scores: max_abs={max_abs} at {max_abs_idx}, max_rel={max_rel}"
    );
    Ok(())
}

#[test]
fn indexer_scores_decode_cutedsl_aot_matches_reference() -> Result<()> {
    let local_heads = 8usize;
    let compressed_len = 8usize;
    let score_scale = 0.25f32;

    let q = patterned_bf16(local_heads * HEAD_DIM, 0.002, -0.001);
    let kv = patterned_bf16(compressed_len * HEAD_DIM, 0.003, 0.001);
    let weights = patterned_bf16(local_heads, 0.02, 0.08);
    let expected = reference_decode(&q, &kv, &weights, local_heads, compressed_len, score_scale);

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let stream: CUstream = ptr::null_mut();

    let got = cutedsl_scores(
        &q_d,
        &kv_d,
        &weights,
        1,
        local_heads,
        compressed_len,
        score_scale,
        stream,
    )?;
    assert_close(&got, &expected);
    Ok(())
}

#[test]
fn indexer_scores_decode_cutedsl_aot_handles_single_compressed_block() -> Result<()> {
    let local_heads = 8usize;
    let compressed_len = 1usize;
    let score_scale = 0.25f32;

    let q = patterned_bf16(local_heads * HEAD_DIM, 0.002, -0.001);
    let kv = patterned_bf16(compressed_len * HEAD_DIM, 0.003, 0.001);
    let weights = patterned_bf16(local_heads, 0.02, 0.08);
    let expected = reference_decode(&q, &kv, &weights, local_heads, compressed_len, score_scale);

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let stream: CUstream = ptr::null_mut();

    let got = cutedsl_scores(
        &q_d,
        &kv_d,
        &weights,
        1,
        local_heads,
        compressed_len,
        score_scale,
        stream,
    )?;
    assert_close(&got, &expected);
    Ok(())
}
