#![cfg(feature = "deepseek-v4")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::{Result, ensure};
use cudarc::driver::sys::CUstream;
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

fn run_topk_prefill(
    scores: &[f32],
    seq_len: usize,
    compressed_len: usize,
    topk: usize,
    ratio: usize,
    offset: usize,
) -> Result<Vec<i32>> {
    let scores_d = DeviceBuffer::from_host(scores)?;
    let mut topk_d = DeviceBuffer::<i32>::zeroed(seq_len * topk)?;
    let stream: CUstream = ptr::null_mut();
    let result = unsafe {
        ffi::deepseek_indexer_topk_prefill_cuda(
            scores_d.as_ptr(),
            topk_d.as_mut_ptr(),
            seq_len as i32,
            compressed_len as i32,
            topk as i32,
            ratio as i32,
            offset as i32,
            stream,
        )
    };
    assert_eq!(result, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;
    topk_d.copy_to_host()
}

fn reference_topk(
    scores: &[f32],
    seq_len: usize,
    compressed_len: usize,
    topk: usize,
    ratio: usize,
    offset: usize,
) -> Vec<i32> {
    let mut out = vec![-1; seq_len * topk];
    let neg_inf = -3.4028234663852886e38f32;
    for token in 0..seq_len {
        let valid = (token + 1) / ratio;
        let mut selected = Vec::with_capacity(topk);
        for candidate in 0..compressed_len {
            let score = if candidate < valid {
                scores[token * compressed_len + candidate]
            } else {
                neg_inf
            };
            let insert_pos = selected
                .iter()
                .position(|&(_, selected_score)| score > selected_score);
            if let Some(pos) = insert_pos {
                selected.insert(pos, (candidate, score));
                selected.truncate(topk);
            } else if selected.len() < topk {
                selected.push((candidate, score));
            }
        }
        for route in 0..topk {
            if let Some(&(candidate, score)) = selected.get(route) {
                out[token * topk + route] = if score > -3.0e38f32 {
                    (offset + candidate) as i32
                } else {
                    -1
                };
            }
        }
    }
    out
}

fn lcg_scores(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 40) & 0xffff) as f32;
            let jitter = bits / 65536.0;
            let bucket = (idx % 17) as f32 * 0.0001;
            jitter + bucket
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU; run on the DSV4 validation host"]
fn indexer_topk_prefill_matches_reference_odd_shape() -> Result<()> {
    let seq_len = 257;
    let compressed_len = 129;
    let topk = 32;
    let ratio = 4;
    let offset = 777;
    let scores = lcg_scores(seq_len * compressed_len, 0x5eed_1234);

    let got = run_topk_prefill(&scores, seq_len, compressed_len, topk, ratio, offset)?;
    let expected = reference_topk(&scores, seq_len, compressed_len, topk, ratio, offset);
    assert_eq!(got, expected);
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; covers the 10k prefill launch/shape from task #27"]
fn indexer_topk_prefill_matches_reference_10k_profile_shape() -> Result<()> {
    let seq_len = 10580;
    let compressed_len = 2645;
    let topk = 32;
    let ratio = 4;
    let offset = seq_len;
    let mut scores = vec![0.0f32; seq_len * compressed_len];
    for token in 0..seq_len {
        for compressed in 0..compressed_len {
            scores[token * compressed_len + compressed] =
                compressed as f32 + token as f32 * 0.000001;
        }
    }

    let got = run_topk_prefill(&scores, seq_len, compressed_len, topk, ratio, offset)?;
    for token in 0..seq_len {
        let valid = (token + 1) / ratio;
        for route in 0..topk {
            let expected = if route < valid.min(topk) {
                (offset + valid - 1 - route) as i32
            } else {
                -1
            };
            assert_eq!(
                got[token * topk + route],
                expected,
                "token={token} route={route} valid={valid}"
            );
        }
    }
    Ok(())
}
