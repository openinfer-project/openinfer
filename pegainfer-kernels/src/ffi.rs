// FFI surface for CUDA/cuBLAS/FlashInfer kernels, split by owning model.
// Public paths are unchanged: `pegainfer_kernels::ffi::<symbol>` resolves via the re-exports below.

// Half type (16-bit float) - same layout as CUDA half. Shared ABI type used by all submodules.
pub type Half = u16;

#[cfg(feature = "kimi-k2")]
mod deepep;
mod deepseek;
#[cfg(feature = "kimi-k2")]
mod kimi;
mod qwen35;
mod shared;
#[cfg(feature = "kimi-k2")]
pub use deepep::*;
pub use deepseek::*;
#[cfg(feature = "kimi-k2")]
pub use kimi::*;
pub use qwen35::*;
pub use shared::*;
