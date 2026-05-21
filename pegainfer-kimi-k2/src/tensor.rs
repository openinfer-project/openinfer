//! Minimal tensor/type vocabulary used by the header modules.

use std::{marker::PhantomData, ops::Range};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bf16,
    F32,
    U8,
    U16,
    U32,
    I32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
    HeadMajor,
    ExpertMajor,
    Paged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape2 {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape3 {
    pub outer: usize,
    pub middle: usize,
    pub inner: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DevicePtr<T> {
    pub addr: u64,
    pub len: usize,
    _marker: PhantomData<T>,
}

impl<T> DevicePtr<T> {
    #[must_use]
    pub const fn new(addr: u64, len: usize) -> Self {
        Self {
            addr,
            len,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorRef<T> {
    pub ptr: DevicePtr<T>,
    pub dtype: DType,
    pub layout: Layout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorMut<T> {
    pub ptr: DevicePtr<T>,
    pub dtype: DType,
    pub layout: Layout,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bf16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct F32;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct U8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct U32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenBatch {
    pub batch_size: usize,
    pub active_tokens: usize,
    pub padded_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TpRank {
    pub rank: usize,
    pub world: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpRank {
    pub rank: usize,
    pub world: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VocabShard {
    pub range: Range<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamHandle(pub u64);

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum HeaderError {
    #[error("shape mismatch: {message}")]
    Shape { message: String },
    #[error("unsupported operator path: {message}")]
    Unsupported { message: String },
}

pub type HeaderResult<T> = Result<T, HeaderError>;
