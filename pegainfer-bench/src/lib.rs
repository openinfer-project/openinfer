//! Model-agnostic kernel benchmarking harness.
//!
//! Every model crate's kernel-report tooling needs the same three things: a
//! CUDA-event timing loop, latency statistics, and accessors over the
//! [`KernelCall`] schedule that the report bins serialize. Those pieces carry no
//! model knowledge, so they live here and each model crate keeps only its own
//! `measure_*` providers, which call into [`measure_loop`].

use anyhow::{Result, anyhow, bail};
use cudarc::driver::sys;
use pegainfer_kernels::tensor::{DeviceContext, DeviceMatrix, GpuWeight, KernelCall, TensorSpec};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct LatencyStats {
    pub iters: u64,
    pub mean_us: f64,
    pub stddev_us: f64,
    pub min_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MeasuredCall {
    pub supported: bool,
    pub reason: Option<String>,
    pub stats: Option<LatencyStats>,
}

impl LatencyStats {
    /// All-zero stats for calls that are counted but deliberately not timed
    /// (e.g. a no-op collective on a single rank).
    pub fn zero(iters: u64) -> Self {
        Self {
            iters,
            mean_us: 0.0,
            stddev_us: 0.0,
            min_us: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
            max_us: 0.0,
        }
    }

    pub fn from_samples(iters: u64, mut samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty() {
            bail!("latency sample set is empty");
        }
        samples.sort_by(f64::total_cmp);
        let mean_us = samples.iter().sum::<f64>() / samples.len() as f64;
        let stddev_us = if samples.len() > 1 {
            let variance = samples
                .iter()
                .map(|sample| {
                    let delta = sample - mean_us;
                    delta * delta
                })
                .sum::<f64>()
                / (samples.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        Ok(Self {
            iters,
            mean_us,
            stddev_us,
            min_us: samples[0],
            p50_us: percentile(&samples, 0.50),
            p95_us: percentile(&samples, 0.95),
            p99_us: percentile(&samples, 0.99),
            max_us: samples[samples.len() - 1],
        })
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Time `launch` over `iters` CUDA-event-bracketed iterations after a 3-shot warmup.
pub fn measure_loop(
    ctx: &DeviceContext,
    iters: u64,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<LatencyStats> {
    if iters == 0 {
        bail!("iters must be greater than zero");
    }
    for _ in 0..3 {
        launch()?;
    }
    ctx.sync()?;
    let start = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        start.record(&ctx.stream)?;
        launch()?;
        end.record(&ctx.stream)?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1_000.0);
    }
    ctx.sync()?;
    LatencyStats::from_samples(iters, samples)
}

/// Stable JSON identity of a [`KernelCall`] — op name plus its input/output/attr shapes.
pub fn bench_key(call: &KernelCall) -> Result<String> {
    Ok(serde_json::to_string(&serde_json::json!({
        "op": call.op,
        "inputs": call.inputs,
        "outputs": call.outputs,
        "attrs": call.attrs,
    }))?)
}

pub fn axis(spec: &TensorSpec, name: &str) -> Result<usize> {
    spec.axes
        .iter()
        .find(|axis| axis.name == name)
        .map(|axis| axis.size)
        .ok_or_else(|| anyhow!("missing axis `{name}` in {}", spec.compact()))
}

pub fn input<'a>(call: &'a KernelCall, name: &str) -> Result<&'a TensorSpec> {
    call.inputs
        .iter()
        .find(|arg| arg.name == name)
        .map(|arg| &arg.spec)
        .ok_or_else(|| anyhow!("{} missing input `{name}`", call.label))
}

pub fn output<'a>(call: &'a KernelCall, name: &str) -> Result<&'a TensorSpec> {
    call.outputs
        .iter()
        .find(|arg| arg.name == name)
        .map(|arg| &arg.spec)
        .ok_or_else(|| anyhow!("{} missing output `{name}`", call.label))
}

pub fn attr_usize(call: &KernelCall, name: &str) -> Result<usize> {
    call.attrs
        .iter()
        .find(|attr| attr.name == name)
        .ok_or_else(|| anyhow!("{} missing attr `{name}`", call.label))?
        .value
        .parse()
        .map_err(|err| anyhow!("{} invalid attr `{name}`: {err}", call.label))
}

/// Zero-initialized `rows × cols` device matrix, sized straight from a kernel shape.
pub fn zero_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    Ok(DeviceMatrix {
        data: ctx.stream.alloc_zeros(rows * cols)?,
        rows,
        cols,
    })
}

pub fn zero_weight<const OUT: usize, const IN: usize>(
    ctx: &DeviceContext,
) -> Result<GpuWeight<OUT, IN>> {
    GpuWeight::from_device_matrix(zero_matrix(ctx, OUT, IN)?)
}
