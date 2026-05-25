#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RequestId(pub(crate) u64);

impl RequestId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}
