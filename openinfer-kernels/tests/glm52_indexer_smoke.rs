//! GLM5.2 DSA indexer kernel smoke tests.
//!
//! Validates that the hand-written and vendored wrapper kernels launch
//! correctly on Hopper (sm_90) and produce expected results for simple
//! inputs. Not an oracle gate — correctness is against simple host-side
//! reference implementations.
//!
//!   cargo test --release -p openinfer-kernels --features glm52 \
//!     --test glm52_indexer_smoke -- --nocapture

#![cfg(feature = "glm52")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::{Result, ensure};
use cudarc::driver::sys::{CUresult, CUstream};
use half::bf16;
use openinfer_kernels::ffi;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) })?;
        Ok(Self { ptr })
    }

    fn from_host<T: Copy>(data: &[T]) -> Result<Self> {
        let bytes = data.len() * size_of::<T>();
        let buf = Self::alloc(bytes)?;
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    buf.ptr,
                    data.as_ptr().cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            })?;
        }
        Ok(buf)
    }

    fn zeroed(bytes: usize) -> Result<Self> {
        let buf = Self::alloc(bytes)?;
        cuda_check(unsafe { cudaMemset(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    fn to_host<T: Copy + Default>(&self, count: usize) -> Result<Vec<T>> {
        let mut data = vec![T::default(); count];
        let bytes = count * size_of::<T>();
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    data.as_mut_ptr().cast::<c_void>(),
                    self.ptr,
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
        }
        Ok(data)
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
        }
    }
}

fn cuda_check(code: i32) -> Result<()> {
    ensure!(code == 0, "CUDA runtime call failed with code {code}");
    Ok(())
}

fn cu_check(r: CUresult) -> Result<()> {
    ensure!(
        r == CUresult::CUDA_SUCCESS,
        "CUDA FFI call failed with code {r:?}"
    );
    Ok(())
}

const STREAM: CUstream = ptr::null_mut();

// ─── indexer cache: quant + pack → gather round-trip ──────────────────────

fn indexer_cache_round_trip() -> Result<()> {
    let tokens = 4;
    let head_dim = 128;
    let cache_block_size = 4;
    let stride = cache_block_size * (head_dim + 4); // 128 fp8 + 4 scale bytes per token
    let cache_bytes = stride; // 1 block

    let k_host: Vec<bf16> = (0..tokens * head_dim)
        .map(|idx| {
            let tok = idx / head_dim;
            let dim = idx % head_dim;
            bf16::from_f32((tok as f32 + 1.0) * 0.1 * (dim as f32 + 1.0))
        })
        .collect();
    let slot_mapping: Vec<i64> = (0..tokens as i64).collect();

    let k_dev = DeviceBuf::from_host(&k_host)?;
    let slot_dev = DeviceBuf::from_host(&slot_mapping)?;
    let cache_dev = DeviceBuf::zeroed(cache_bytes)?;

    cu_check(unsafe {
        ffi::glm52_indexer_k_quant_and_cache_cuda(
            k_dev.ptr as *const ffi::Half,
            cache_dev.ptr as *mut u8,
            slot_dev.ptr as *const i64,
            tokens as i32,
            head_dim as i32,
            128, // quant_block_size
            cache_block_size as i32,
            stride as i64,
            0, // F32 scale
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    // Gather back
    let block_table: Vec<i32> = vec![0];
    let cu_seq_lens: Vec<i32> = vec![0, tokens as i32];
    let block_dev = DeviceBuf::from_host(&block_table)?;
    let cu_dev = DeviceBuf::from_host(&cu_seq_lens)?;
    let dst_k = DeviceBuf::zeroed(tokens * head_dim)?;
    let dst_scale = DeviceBuf::zeroed(tokens * 4)?;

    cu_check(unsafe {
        ffi::glm52_indexer_k_gather_quant_cache_cuda(
            cache_dev.ptr as *const u8,
            dst_k.ptr as *mut u8,
            dst_scale.ptr as *mut u8,
            block_dev.ptr as *const i32,
            cu_dev.ptr as *const i32,
            1, // batch_size
            1, // num_blocks_per_seq
            tokens as i32,
            head_dim as i32,
            128, // quant_block_size
            cache_block_size as i32,
            stride as i64,
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let scales: Vec<f32> = dst_scale.to_host(tokens)?;
    for (i, &s) in scales.iter().enumerate() {
        ensure!(s > 0.0, "indexer cache: scale[{}] = {}, expected > 0", i, s);
    }

    eprintln!(
        "indexer_cache_round_trip: {} scales all positive, OK",
        tokens
    );
    Ok(())
}

// ─── local_topk_to_slots ──────────────────────────────────────────────────

fn local_topk_to_slots_basic() -> Result<()> {
    // 1 token, topk=4, block_size=2, 2 pages
    // block_table = [10, 20], offsets = [0,1,2,3] -> slots [20,21,40,41]
    let offsets: Vec<i32> = vec![0, 1, 2, 3];
    let block_table: Vec<i32> = vec![10, 20];
    let seq_lens: Vec<i32> = vec![4];

    let offsets_dev = DeviceBuf::from_host(&offsets)?;
    let block_dev = DeviceBuf::from_host(&block_table)?;
    let seq_dev = DeviceBuf::from_host(&seq_lens)?;
    let slots_dev = DeviceBuf::zeroed(4 * size_of::<i32>())?;
    let lens_dev = DeviceBuf::zeroed(1 * size_of::<i32>())?;

    cu_check(unsafe {
        ffi::glm52_indexer_local_topk_to_slots_cuda(
            slots_dev.ptr as *mut i32,
            lens_dev.ptr as *mut i32,
            offsets_dev.ptr as *const i32,
            4, // local_topk_stride
            seq_dev.ptr as *const i32,
            block_dev.ptr as *const i32,
            2, // block_table_stride
            2, // block_table_cols
            2, // block_size
            4, // topk
            1, // num_tokens
            1, // has_seq_lens
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let slots: Vec<i32> = slots_dev.to_host(4)?;
    let lens: Vec<i32> = lens_dev.to_host(1)?;

    assert_eq!(lens[0], 4, "topk_lens[0]");
    assert_eq!(slots[0], 20, "slot[0]");
    assert_eq!(slots[1], 21, "slot[1]");
    assert_eq!(slots[2], 40, "slot[2]");
    assert_eq!(slots[3], 41, "slot[3]");

    eprintln!(
        "local_topk_to_slots_basic: slots = {:?}, lens = {:?}, OK",
        slots, lens
    );
    Ok(())
}

fn local_topk_to_slots_invalid_offset() -> Result<()> {
    let offsets: Vec<i32> = vec![-1, 0];
    let block_table: Vec<i32> = vec![5];
    let seq_lens: Vec<i32> = vec![2];

    let offsets_dev = DeviceBuf::from_host(&offsets)?;
    let block_dev = DeviceBuf::from_host(&block_table)?;
    let seq_dev = DeviceBuf::from_host(&seq_lens)?;
    let slots_dev = DeviceBuf::zeroed(2 * size_of::<i32>())?;
    let lens_dev = DeviceBuf::zeroed(1 * size_of::<i32>())?;

    cu_check(unsafe {
        ffi::glm52_indexer_local_topk_to_slots_cuda(
            slots_dev.ptr as *mut i32,
            lens_dev.ptr as *mut i32,
            offsets_dev.ptr as *const i32,
            2, // stride
            seq_dev.ptr as *const i32,
            block_dev.ptr as *const i32,
            1, // block_table_stride
            1, // block_table_cols
            2, // block_size
            2, // topk
            1, // num_tokens
            1, // has_seq_lens
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let slots: Vec<i32> = slots_dev.to_host(2)?;
    let lens: Vec<i32> = lens_dev.to_host(1)?;

    assert_eq!(slots[0], -1, "invalid offset -> -1");
    assert_eq!(slots[1], 10, "offset 0, page 5, bs 2 -> 10");
    assert_eq!(lens[0], 1, "1 valid");

    eprintln!(
        "local_topk_to_slots_invalid_offset: slots = {:?}, lens = {:?}, OK",
        slots, lens
    );
    Ok(())
}

// ─── Hadamard ─────────────────────────────────────────────────────────────

fn hadamard_correctness() -> Result<()> {
    let head_dim = 128;
    let input: Vec<bf16> = vec![bf16::from_f32(1.0); head_dim];

    let in_dev = DeviceBuf::from_host(&input)?;
    let out_dev = DeviceBuf::zeroed(head_dim * size_of::<bf16>())?;

    cu_check(unsafe {
        ffi::glm52_indexer_hadamard_bf16_cuda(
            in_dev.ptr as *const ffi::Half,
            out_dev.ptr as *mut ffi::Half,
            1, // tokens
            head_dim as i32,
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let output: Vec<bf16> = out_dev.to_host(head_dim)?;

    // Row 0 of H_128 is all +1, so output[0] = sum(1..128) * (1/sqrt(128)) = sqrt(128)
    let expected = (head_dim as f32).sqrt();
    let got = output[0].to_f32();
    ensure!(
        (got - expected).abs() < 0.1,
        "Hadamard[0] = {}, expected ~{}",
        got,
        expected
    );

    // Row 1 has half +1 half -1, so output[1] ≈ 0
    let got1 = output[1].to_f32();
    ensure!(got1.abs() < 0.1, "Hadamard[1] = {}, expected ~0", got1);

    eprintln!(
        "hadamard_correctness: [0]={:.4} (exp {:.4}), [1]={:.4} (exp 0), OK",
        got, expected, got1
    );
    Ok(())
}

// ─── FlashInfer top-k ─────────────────────────────────────────────────────

fn flashinfer_topk_basic() -> Result<()> {
    let max_len = 2048;
    let top_k = 4;

    // value = index, so top-k are indices 2044..2047
    let logits: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
    let lengths: Vec<i32> = vec![max_len as i32];

    let logits_dev = DeviceBuf::from_host(&logits)?;
    let lengths_dev = DeviceBuf::from_host(&lengths)?;
    let indices_dev = DeviceBuf::zeroed(top_k * size_of::<i32>())?;
    let values_dev = DeviceBuf::zeroed(top_k * size_of::<f32>())?;

    let scratch_bytes = unsafe { ffi::glm52_flashinfer_topk_2048_row_states_bytes_cuda() };
    let scratch_dev = DeviceBuf::zeroed(scratch_bytes)?;

    cu_check(unsafe {
        ffi::glm52_flashinfer_topk_2048_cuda(
            logits_dev.ptr as *const f32,
            indices_dev.ptr as *mut i32,
            values_dev.ptr as *mut f32,
            lengths_dev.ptr as *const i32,
            1, // num_rows
            top_k as i32,
            max_len as i32,
            scratch_dev.ptr as *mut u8,
            STREAM,
        )
    })?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let indices: Vec<i32> = indices_dev.to_host(top_k)?;
    let mut sorted = indices.to_vec();
    sorted.sort();
    assert_eq!(sorted, vec![2044, 2045, 2046, 2047], "top-k indices");

    eprintln!("flashinfer_topk_basic: indices = {:?} (sorted), OK", sorted);
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[test]
fn test_indexer_cache_round_trip() {
    indexer_cache_round_trip().expect("indexer cache round-trip");
}

#[test]
fn test_local_topk_to_slots_basic() {
    local_topk_to_slots_basic().expect("local_topk_to_slots basic");
}

#[test]
fn test_local_topk_to_slots_invalid() {
    local_topk_to_slots_invalid_offset().expect("local_topk_to_slots invalid");
}

#[test]
fn test_hadamard_correctness() {
    hadamard_correctness().expect("Hadamard");
}

#[test]
fn test_flashinfer_topk_basic() {
    flashinfer_topk_basic().expect("FlashInfer top-k");
}

// ─── DeepGEMM paged MQA logits: JIT compile + launch smoke test ────────────
//
// This is the first DeepGEMM JIT kernel call in the codebase. The test
// verifies that the torch-free wrapper can:
//   1. Initialize the JIT compiler (needs OPENINFER_DEEPGEMM_ROOT + CUDA_HOME)
//   2. Compile the paged MQA logits kernel via nvcc JIT
//   3. Launch both the metadata and logits kernels without crashing
//
// It does NOT validate logits correctness — that requires an oracle gate
// with HF reference outputs. The smoke test only checks launch success.

fn deepgemm_env_ready() -> bool {
    std::env::var("OPENINFER_DEEPGEMM_ROOT").is_ok() && std::env::var("CUDA_HOME").is_ok()
}

fn deepgemm_paged_mqa_launch() -> Result<()> {
    if !deepgemm_env_ready() {
        eprintln!("skip: set OPENINFER_DEEPGEMM_ROOT + CUDA_HOME to run DeepGEMM MQA test");
        return Ok(());
    }

    let batch_size = 1_i32;
    let next_n = 1_i32;
    let num_heads = 4_i32;
    let head_dim = 128_i32;
    let block_kv = 128_i32;
    let num_kv_blocks = 16_i32; // 2048 tokens
    let num_sms = 132_i32; // H200 has 132 SMs
    let logits_stride = 256_i32; // split_kv=256
    let block_table_stride = num_kv_blocks;

    // context_lens: each batch has 2048 KV tokens
    let context_lens_host = vec![2048_i32; batch_size as usize];
    let context_lens = DeviceBuf::from_host(&context_lens_host)?;

    // schedule_metadata: aligned_batch_size = 32 (batch=1 → align to 32)
    let sched_meta_len = 32_i32 as usize; // non-varlen: aligned_batch_size
    let schedule_metadata = DeviceBuf::zeroed(sched_meta_len * std::mem::size_of::<i32>())?;

    // Metadata kernel launch
    let r = unsafe {
        ffi::glm52_deepgemm_paged_mqa_metadata_cuda(
            context_lens.ptr as *mut i32,
            schedule_metadata.ptr as *mut i32,
            batch_size,
            next_n,
            block_kv,
            num_sms,
            false, // is_context_lens_2d
            false, // is_varlen
            std::ptr::null(),
            STREAM,
        )
    };
    cu_check(r)?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    // Logits kernel launch
    // q: [batch_size, next_n, num_heads, head_dim] fp8 = 1*1*4*128 = 512 bytes
    let q_bytes = (batch_size * next_n * num_heads * head_dim) as usize;
    let q = DeviceBuf::zeroed(q_bytes)?;

    // kv_cache: [num_kv_blocks, block_kv, head_dim] fp8 = 16*128*128 = 262144 bytes
    let kv_bytes = (num_kv_blocks * block_kv * head_dim) as usize;
    let kv_cache = DeviceBuf::zeroed(kv_bytes)?;

    // kv_cache_scales: [num_kv_blocks, block_kv] f32 = 16*128*4 = 8192 bytes
    let kv_scales_bytes = (num_kv_blocks * block_kv) as usize * std::mem::size_of::<f32>();
    let kv_cache_scales = DeviceBuf::zeroed(kv_scales_bytes)?;

    // weights: [batch_size * next_n, num_heads] fp8 = 1*4 = 4 bytes
    let weights_bytes = (batch_size * next_n * num_heads) as usize;
    let weights = DeviceBuf::zeroed(weights_bytes)?;

    // logits: [batch_size, logits_stride] bf16 = 1*256*2 = 512 bytes
    let logits_bytes = (batch_size * logits_stride) as usize * std::mem::size_of::<bf16>();
    let logits = DeviceBuf::zeroed(logits_bytes)?;

    // block_table: [batch_size, block_table_stride] i32
    let bt_bytes = (batch_size * block_table_stride) as usize * std::mem::size_of::<i32>();
    let block_table_host: Vec<i32> = (0..num_kv_blocks).collect();
    let block_table = DeviceBuf::from_host(&block_table_host)?;

    let r = unsafe {
        ffi::glm52_deepgemm_paged_mqa_logits_cuda(
            q.ptr,
            kv_cache.ptr,
            kv_cache_scales.ptr as *const f32,
            weights.ptr,
            context_lens.ptr as *const i32,
            logits.ptr,
            block_table.ptr as *const i32,
            std::ptr::null(),
            schedule_metadata.ptr as *mut i32,
            batch_size,
            next_n,
            num_heads,
            head_dim,
            num_kv_blocks,
            block_kv,
            false, // is_context_lens_2d
            false, // is_varlen
            logits_stride,
            block_table_stride,
            num_sms,
            1, // q_elem_size (fp8)
            1, // kv_elem_size (fp8)
            1, // weights_elem_size (fp8)
            4, // kv_scales_elem_size (f32)
            STREAM,
        )
    };
    cu_check(r)?;
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    // Verify logits are all zero (since all inputs are zeroed)
    let logits_host: Vec<bf16> = logits.to_host((batch_size * logits_stride) as usize)?;
    let all_zero = logits_host.iter().all(|v| v.to_f32() == 0.0);
    ensure!(
        all_zero,
        "DeepGEMM MQA logits should be all zero for zeroed inputs"
    );

    Ok(())
}

#[test]
fn test_deepgemm_paged_mqa_launch() {
    deepgemm_paged_mqa_launch().expect("DeepGEMM paged MQA logits launch");
}
