use crate::tensor::{Shape2, Shape3, TensorMut, TensorRef};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor1Ref<T> {
    pub tensor: TensorRef<T>,
    pub len: usize,
}

impl<T> Tensor1Ref<T> {
    #[must_use]
    pub const fn new(tensor: TensorRef<T>, len: usize) -> Self {
        Self { tensor, len }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor1Mut<T> {
    pub tensor: TensorMut<T>,
    pub len: usize,
}

impl<T> Tensor1Mut<T> {
    #[must_use]
    pub const fn new(tensor: TensorMut<T>, len: usize) -> Self {
        Self { tensor, len }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor2Ref<T> {
    pub tensor: TensorRef<T>,
    pub shape: Shape2,
}

impl<T> Tensor2Ref<T> {
    #[must_use]
    pub const fn new(tensor: TensorRef<T>, shape: Shape2) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor2Mut<T> {
    pub tensor: TensorMut<T>,
    pub shape: Shape2,
}

impl<T> Tensor2Mut<T> {
    #[must_use]
    pub const fn new(tensor: TensorMut<T>, shape: Shape2) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor3Ref<T> {
    pub tensor: TensorRef<T>,
    pub shape: Shape3,
}

impl<T> Tensor3Ref<T> {
    #[must_use]
    pub const fn new(tensor: TensorRef<T>, shape: Shape3) -> Self {
        Self { tensor, shape }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor3Mut<T> {
    pub tensor: TensorMut<T>,
    pub shape: Shape3,
}

impl<T> Tensor3Mut<T> {
    #[must_use]
    pub const fn new(tensor: TensorMut<T>, shape: Shape3) -> Self {
        Self { tensor, shape }
    }
}
