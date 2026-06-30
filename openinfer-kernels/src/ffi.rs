// FFI surface for CUDA/cuBLAS/FlashInfer kernels, split by owning model.
// Public paths are unchanged: `openinfer_kernels::ffi::<symbol>` resolves via the re-exports below.

// Half type (16-bit float) - same layout as CUDA half. Shared ABI type used by all submodules.
pub type Half = u16;

#[cfg(feature = "moe")]
mod deepep;
mod deepseek;
#[cfg(feature = "deepseek-v2-lite")]
mod deepseek_v2_lite;
#[cfg(feature = "glm52")]
mod glm52;
#[cfg(feature = "kimi-k2")]
mod kimi;
mod lora;
mod qwen35;
mod shared;
#[cfg(feature = "moe")]
pub use deepep::*;
pub use deepseek::*;
#[cfg(feature = "deepseek-v2-lite")]
pub use deepseek_v2_lite::*;
#[cfg(feature = "glm52")]
pub use glm52::*;
#[cfg(feature = "kimi-k2")]
pub use kimi::*;
pub use lora::*;
pub use qwen35::*;
pub use shared::*;
