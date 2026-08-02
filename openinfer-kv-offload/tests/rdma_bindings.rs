use rdma_mummy_sys::ibv_context;
use rdma_mummy_sys::ibv_mr;

#[test]
fn generated_verbs_structs_are_not_opaque() {
    assert!(std::mem::offset_of!(ibv_context, ops) < std::mem::size_of::<ibv_context>());
    assert!(std::mem::offset_of!(ibv_mr, lkey) < std::mem::size_of::<ibv_mr>());
}
