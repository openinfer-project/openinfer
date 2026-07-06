//! jz-38-only accept-rate probe for sampled-verify speculative decoding
//! (never merge this branch).
//!
//! Dumps the raw ingredients of the A/B accept-rate question — per decode
//! step the target's full logits row, per draft round the block logits and
//! per-position Markov bias rows — so an offline script can compute, for any
//! (temperature, top_p) grid:
//!   A: p'(draft_k)          (greedy draft + sampled verify)
//!   B: sum_x min(p', q')    (full rejection sampling headroom)
//!
//! Enabled iff [`DIR`] exists on the node; the run script mkdirs it. All
//! writes are synchronous and slow — the probe measures accept rates, not
//! wall clock.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const DIR: &str = "/root/develop/xingming/abprobe/dump";

pub(crate) fn dir() -> Option<&'static Path> {
    static D: OnceLock<Option<PathBuf>> = OnceLock::new();
    D.get_or_init(|| {
        let p = PathBuf::from(DIR);
        p.is_dir().then_some(p)
    })
    .as_deref()
}

pub(crate) fn enabled() -> bool {
    dir().is_some()
}

/// Write one full-vocab logits row as raw little-endian bf16 bytes.
pub(crate) fn write_bf16(name: &str, row: &[half::bf16]) {
    let Some(dir) = dir() else { return };
    // bf16 is a transparent u16 wrapper; the raw-byte view is the file format.
    let bytes = unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), size_of_val(row)) };
    if let Err(err) = std::fs::write(dir.join(name), bytes) {
        log::warn!("abprobe: write {name}: {err}");
    }
}

/// Append one JSON line to the probe manifest.
pub(crate) fn manifest(line: &str) {
    let Some(dir) = dir() else { return };
    let open = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("manifest.jsonl"));
    match open {
        Ok(mut f) => {
            let _ = writeln!(f, "{line}");
        }
        Err(err) => log::warn!("abprobe: manifest append: {err}"),
    }
}
