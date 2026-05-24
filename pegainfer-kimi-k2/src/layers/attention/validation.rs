use crate::{
    layers::attention::{Tensor1Ref, Tensor2Mut, Tensor2Ref, Tensor3Mut, Tensor3Ref},
    tensor::{Bf16, DType, F32, HeaderError, HeaderResult, TokenBatch, U32},
};

pub(super) fn validate_token_batch(batch: TokenBatch) -> HeaderResult<()> {
    if batch.batch_size == 0 {
        return shape_err("batch_size must be non-zero");
    }
    if batch.active_tokens == 0 {
        return shape_err("active_tokens must be non-zero");
    }
    if batch.padded_tokens < batch.active_tokens {
        return shape_err("padded_tokens must be >= active_tokens");
    }
    Ok(())
}

pub(super) fn validate_decode_batch(batch: TokenBatch) -> HeaderResult<()> {
    validate_token_batch(batch)?;
    if batch.active_tokens != batch.batch_size {
        return shape_err("decode batch must have one active token per request");
    }
    if batch.padded_tokens < batch.batch_size {
        return shape_err("decode padded_tokens must be >= batch_size");
    }
    Ok(())
}

pub(super) fn expect_1d_ref<T>(name: &str, tensor: Tensor1Ref<T>, len: usize) -> HeaderResult<()> {
    if tensor.len != len {
        return shape_err(format!("{name} expected len {len}, got {}", tensor.len));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

pub(super) fn expect_2d_ref<T>(
    name: &str,
    tensor: Tensor2Ref<T>,
    rows: usize,
    cols: usize,
) -> HeaderResult<()> {
    if tensor.shape.rows != rows || tensor.shape.cols != cols {
        return shape_err(format!(
            "{name} expected shape [{rows}, {cols}], got [{}, {}]",
            tensor.shape.rows, tensor.shape.cols
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

pub(super) fn expect_2d_mut<T>(
    name: &str,
    tensor: Tensor2Mut<T>,
    rows: usize,
    cols: usize,
) -> HeaderResult<()> {
    if tensor.shape.rows != rows || tensor.shape.cols != cols {
        return shape_err(format!(
            "{name} expected shape [{rows}, {cols}], got [{}, {}]",
            tensor.shape.rows, tensor.shape.cols
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

pub(super) fn expect_3d_ref<T>(
    name: &str,
    tensor: Tensor3Ref<T>,
    outer: usize,
    middle: usize,
    inner: usize,
) -> HeaderResult<()> {
    if tensor.shape.outer != outer || tensor.shape.middle != middle || tensor.shape.inner != inner {
        return shape_err(format!(
            "{name} expected shape [{outer}, {middle}, {inner}], got [{}, {}, {}]",
            tensor.shape.outer, tensor.shape.middle, tensor.shape.inner
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

pub(super) fn expect_3d_mut<T>(
    name: &str,
    tensor: Tensor3Mut<T>,
    outer: usize,
    middle: usize,
    inner: usize,
) -> HeaderResult<()> {
    if tensor.shape.outer != outer || tensor.shape.middle != middle || tensor.shape.inner != inner {
        return shape_err(format!(
            "{name} expected shape [{outer}, {middle}, {inner}], got [{}, {}, {}]",
            tensor.shape.outer, tensor.shape.middle, tensor.shape.inner
        ));
    }
    expect_dtype(name, tensor.tensor.dtype, dtype_for::<T>())
}

fn dtype_for<T>() -> DType {
    let name = std::any::type_name::<T>();
    if name == std::any::type_name::<Bf16>() {
        DType::Bf16
    } else if name == std::any::type_name::<F32>() {
        DType::F32
    } else if name == std::any::type_name::<U32>() {
        DType::U32
    } else {
        DType::U8
    }
}

pub(super) fn expect_dtype(name: &str, got: DType, expected: DType) -> HeaderResult<()> {
    if got != expected {
        return shape_err(format!("{name} expected dtype {expected:?}, got {got:?}"));
    }
    Ok(())
}

pub(super) fn shape_err<T>(message: impl Into<String>) -> HeaderResult<T> {
    Err(HeaderError::Shape {
        message: message.into(),
    })
}

pub(super) fn unsupported<T>(message: impl Into<String>) -> HeaderResult<T> {
    Err(HeaderError::Unsupported {
        message: message.into(),
    })
}
