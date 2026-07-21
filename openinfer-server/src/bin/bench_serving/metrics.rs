//! Numeric summarization: percentiles, duration/count stats, token traces.

use std::time::Duration;

use crate::report::CountStats;
use crate::report::DurationStats;
use crate::report::GeneratedTokenTrace;

pub(crate) fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub(crate) fn percentiles(
    sorted: &[Duration],
) -> (Duration, Duration, Duration, Duration, Duration) {
    assert!(!sorted.is_empty());
    let n = sorted.len();
    let sum: Duration = sorted.iter().sum();
    let avg = sum / n as u32;
    let p = |pct: f64| sorted[((pct / 100.0) * (n - 1) as f64).round() as usize];
    (avg, p(50.0), p(95.0), p(99.0), sorted[n - 1])
}

pub(crate) fn summarize_durations(samples: &[Duration]) -> DurationStats {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let (avg, p50, p95, p99, max) = percentiles(&sorted);
    DurationStats {
        avg_ms: dur_ms(avg),
        p50_ms: dur_ms(p50),
        p95_ms: dur_ms(p95),
        p99_ms: dur_ms(p99),
        max_ms: dur_ms(max),
        samples: sorted.len(),
    }
}

pub(crate) fn summarize_counts(samples: &[usize]) -> CountStats {
    assert!(!samples.is_empty());
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    let sum: usize = samples.iter().sum();
    CountStats {
        min,
        max,
        avg: sum as f64 / samples.len() as f64,
        samples: samples.len(),
    }
}

pub(crate) fn aggregate_tok_s(tokens: usize, total: Duration) -> Option<f64> {
    if tokens == 0 || total.is_zero() {
        None
    } else {
        Some(tokens as f64 / total.as_secs_f64())
    }
}

pub(crate) fn generated_token_hash(tokens: &[u32]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

pub(crate) fn generated_token_trace(tokens: &[u32]) -> GeneratedTokenTrace {
    GeneratedTokenTrace {
        hash: generated_token_hash(tokens),
        prefix: tokens.iter().copied().take(16).collect(),
        len: tokens.len(),
    }
}
