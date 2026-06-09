use anyhow::{Result, anyhow};
use cudarc::driver::CudaSlice;
use half::bf16;

use super::*;
use crate::tensor::*;

fn bf16_vec(data: &[f32]) -> Vec<bf16> {
    data.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn rms_norm_reference(x: &[bf16], weight: &[bf16], eps: f32, offset: bool) -> Vec<f32> {
    let sum_sq: f32 = x
        .iter()
        .map(|value| {
            let v = value.to_f32();
            v * v
        })
        .sum();
    let inv_rms = 1.0 / ((sum_sq / x.len() as f32) + eps).sqrt();

    x.iter()
        .zip(weight.iter())
        .map(|(value, weight)| {
            let normed = bf16::from_f32(value.to_f32() * inv_rms).to_f32();
            let scale = if offset {
                1.0 + weight.to_f32()
            } else {
                weight.to_f32()
            };
            bf16::from_f32(normed * scale).to_f32()
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {} expected {} got {} (tol {})",
            idx,
            expected,
            actual,
            tol
        );
    }
}

#[test]
fn test_gemv() -> Result<()> {
    let ctx = DeviceContext::new()?;

    // A = [[1, 2, 3], [4, 5, 6]] (2x3) row-major
    // x = [1, 2, 3]
    // y = A @ x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    let a_data = bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x_data = bf16_vec(&[1.0, 2.0, 3.0]);

    let a = DeviceMatrix::from_host(&ctx, &a_data, 2, 3)?;
    let x = DeviceVec::from_host(&ctx, &x_data)?;
    let y = linear(&ctx, &x, &a)?;

    let result = y.to_host(&ctx)?;
    assert!(
        (result[0] - 14.0).abs() < 0.1,
        "Expected 14, got {}",
        result[0]
    );
    assert!(
        (result[1] - 32.0).abs() < 0.1,
        "Expected 32, got {}",
        result[1]
    );

    Ok(())
}

#[test]
fn test_argmax() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let x = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 9.0, 3.0, 8.0]))?;
    let token = argmax(&ctx, &x)?;
    assert_eq!(token, 1, "Expected argmax index 1, got {}", token);
    Ok(())
}

#[test]
fn test_rms_norm() -> Result<()> {
    let ctx = DeviceContext::new()?;

    let x_host = bf16_vec(&[1.0, 2.0, 3.0, 4.0]);
    let w_host = bf16_vec(&[1.0, 1.0, 1.0, 1.0]);
    let x = DeviceVec::from_host(&ctx, &x_host)?;
    let w = DeviceVec::from_host(&ctx, &w_host)?;
    let out = rms_norm(&ctx, &x, &w, 1e-6)?;

    let result = out.to_host(&ctx)?;
    let expected = rms_norm_reference(&x_host, &w_host, 1e-6, false);
    assert_close(&result, &expected, 0.01);

    Ok(())
}

#[test]
fn test_rms_norm_batch_multi_tile() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let hidden_dim = 260;
    let seq_len = 2;

    let x_host_f32: Vec<f32> = (0..hidden_dim * seq_len)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.25)
        .collect();
    let w_host_f32: Vec<f32> = (0..hidden_dim)
        .map(|idx| 0.5 + (idx % 11) as f32 * 0.0625)
        .collect();
    let x_host = bf16_vec(&x_host_f32);
    let w_host = bf16_vec(&w_host_f32);

    let x = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&x_host)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?,
        hidden_dim,
        seq_len,
    };
    let weight = DeviceVec::from_host(&ctx, &w_host)?;
    let mut out = HiddenStates::zeros(&ctx, hidden_dim, seq_len)?;
    rms_norm_batch_into(&ctx, &x, &weight, 1e-6, &mut out);

    let result = ctx
        .stream
        .clone_dtoh(&out.data)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
    ctx.sync()?;
    let result: Vec<f32> = result.iter().map(|value| value.to_f32()).collect();

    let mut expected = Vec::with_capacity(hidden_dim * seq_len);
    for row in 0..seq_len {
        let start = row * hidden_dim;
        expected.extend(rms_norm_reference(
            &x_host[start..start + hidden_dim],
            &w_host,
            1e-6,
            false,
        ));
    }
    assert_close(&result, &expected, 0.02);

    Ok(())
}

#[test]
fn test_rms_norm_offset() -> Result<()> {
    let ctx = DeviceContext::new()?;

    let x_host = bf16_vec(&[-2.0, -0.5, 0.25, 1.5, 3.0, 0.75, -1.25]);
    let w_host = bf16_vec(&[0.0, 0.5, -0.25, 0.125, 1.0, -0.5, 0.25]);
    let x = DeviceVec::from_host(&ctx, &x_host)?;
    let w = DeviceVec::from_host(&ctx, &w_host)?;
    let mut out = DeviceVec::zeros(&ctx, x_host.len())?;
    rms_norm_offset_into(&ctx, &x, &w, 1e-6, &mut out)?;

    let result = out.to_host(&ctx)?;
    let expected = rms_norm_reference(&x_host, &w_host, 1e-6, true);
    assert_close(&result, &expected, 0.02);

    Ok(())
}

#[test]
fn test_embedding_variants() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let embed = DeviceMatrix::from_host(
        &ctx,
        &bf16_vec(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]),
        3,
        4,
    )?;

    let decode_meta = ctx.stream.clone_htod(&[1_u32])?;
    let mut decode_out = DeviceVec::zeros(&ctx, 4)?;
    embedding_decode_into(&ctx, &embed, &decode_meta, &mut decode_out)?;
    let decode_host = decode_out.to_host(&ctx)?;
    assert!(
        (decode_host[0] - 5.0).abs() < 0.01,
        "Expected 5.0, got {}",
        decode_host[0]
    );
    assert!(
        (decode_host[3] - 8.0).abs() < 0.01,
        "Expected 8.0, got {}",
        decode_host[3]
    );

    let token_ids = ctx.stream.clone_htod(&[2_u32, 0_u32])?;
    let mut batch_out = HiddenStates::zeros(&ctx, 4, 2)?;
    embedding_batch(&ctx, &embed, &token_ids, &mut batch_out)?;
    let batch_host = ctx.stream.clone_dtoh(&batch_out.data)?;
    ctx.sync()?;
    assert!(
        (batch_host[0].to_f32() - 9.0).abs() < 0.01,
        "Expected 9.0, got {}",
        batch_host[0]
    );
    assert!(
        (batch_host[3].to_f32() - 12.0).abs() < 0.01,
        "Expected 12.0, got {}",
        batch_host[3]
    );
    assert!(
        (batch_host[4].to_f32() - 1.0).abs() < 0.01,
        "Expected 1.0, got {}",
        batch_host[4]
    );
    assert!(
        (batch_host[7].to_f32() - 4.0).abs() < 0.01,
        "Expected 4.0, got {}",
        batch_host[7]
    );

    Ok(())
}

#[test]
fn test_gpu_sample() -> Result<()> {
    let ctx = DeviceContext::new()?;

    // Create logits with a clear winner at index 2 (highest logit)
    // but with temperature sampling, other tokens have a chance
    let logits_data = bf16_vec(&[1.0, 2.0, 10.0, 1.5, 0.5]);
    let logits = DeviceVec::from_host(&ctx, &logits_data)?;
    let mut probs: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(5)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;
    let mut top1_value: CudaSlice<half::bf16> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;
    let mut row_states: CudaSlice<u8> = ctx
        .stream
        .alloc_zeros(crate::ops::flashinfer_topk_row_states_bytes())
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    // Test 1: With very low temperature (near-greedy), should pick token 2
    let params = crate::sampler::SamplingParams {
        temperature: 0.01,
        top_k: -1,
        top_p: 1.0,
        ..Default::default()
    };
    let token = gpu_sample(
        &ctx,
        &logits,
        &mut probs,
        &mut top1_value,
        &mut row_states,
        &params,
        0.5,
    )?;
    assert_eq!(token, 2, "near-greedy should pick index 2 (highest logit)");

    // Test 2: Pure temperature sampling stays valid after switching to the
    // FlashInfer backend. Exact RNG semantics differ from the legacy kernel.
    let params = crate::sampler::SamplingParams {
        temperature: 1.0,
        top_k: -1,
        top_p: 1.0,
        ..Default::default()
    };
    let token = gpu_sample(
        &ctx,
        &logits,
        &mut probs,
        &mut top1_value,
        &mut row_states,
        &params,
        0.0,
    )?;
    assert!(token < 5, "sampled token should stay in bounds");

    // Test 3: top_k=1 should always pick the highest
    let params = crate::sampler::SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        ..Default::default()
    };
    let token = gpu_sample(
        &ctx,
        &logits,
        &mut probs,
        &mut top1_value,
        &mut row_states,
        &params,
        0.5,
    )?;
    assert_eq!(token, 2, "top_k=1 should pick highest probability token");

    Ok(())
}
