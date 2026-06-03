// FFI surface for CUDA/cuBLAS/FlashInfer kernels, split by owning model.
// Public paths are unchanged: `pegainfer_kernels::ffi::<symbol>` resolves via the re-exports below.

// Half type (16-bit float) - same layout as CUDA half. Shared ABI type used by all submodules.
pub type Half = u16;

mod deepseek;
#[cfg(feature = "kimi-k2")]
mod kimi;
mod lora;
mod qwen35;
mod shared;
pub use deepseek::*;
#[cfg(feature = "kimi-k2")]
pub use kimi::*;
pub use lora::*;
pub use qwen35::*;
pub use shared::*;
