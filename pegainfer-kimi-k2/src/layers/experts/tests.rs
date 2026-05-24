use super::*;
use crate::tensor::DevicePtr;

#[test]
fn ep8_meta_matches_kimi_shapes() {
    let ep_rank = EpRank { rank: 3, world: 8 };
    let w1 = CompressedTensorsInt4Meta::ep8(
        ExpertLinearRole::W1Gate,
        ep_rank,
        Int4NibbleOrder::LowThenHigh,
    );
    assert_eq!(w1.local_experts, 48);
    assert_eq!(w1.local_expert_offset, 144);
    assert_eq!(w1.logical_shape.rows, 2048);
    assert_eq!(w1.logical_shape.cols, 7168);
    assert_eq!(w1.packed_shape.inner, 3584);
    assert_eq!(w1.scale_shape.inner, 224);
    w1.validate().unwrap();

    let w2 = CompressedTensorsInt4Meta::ep8(
        ExpertLinearRole::W2Down,
        ep_rank,
        Int4NibbleOrder::LowThenHigh,
    );
    assert_eq!(w2.logical_shape.rows, 7168);
    assert_eq!(w2.logical_shape.cols, 2048);
    assert_eq!(w2.packed_shape.inner, 1024);
    assert_eq!(w2.scale_shape.inner, 64);
    w2.validate().unwrap();
}

#[test]
fn route_layout_allows_multi_sequence_batches() {
    let route = ExpertRouteLayout {
        batch: TokenBatch {
            batch_size: 4,
            active_tokens: 17,
            padded_tokens: 32,
        },
        local_expert_offset: 0,
        local_experts: 48,
        routed_tokens: 17 * KIMI_K2_TOPK,
        topk_indices: tref_u32(17 * KIMI_K2_TOPK, Layout::RowMajor),
        topk_weights: tref_f32(17 * KIMI_K2_TOPK, Layout::RowMajor),
        expert_indptr: tmut_u32(49, Layout::ExpertMajor),
        route_to_token: tmut_u32(17 * KIMI_K2_TOPK, Layout::ExpertMajor),
        route_to_topk_slot: tmut_u32(17 * KIMI_K2_TOPK, Layout::ExpertMajor),
    };

    route.validate().unwrap();
}

fn tref_u32(len: usize, layout: Layout) -> TensorRef<U32> {
    TensorRef {
        ptr: DevicePtr::new(1, len),
        dtype: DType::U32,
        layout,
    }
}

fn tref_f32(len: usize, layout: Layout) -> TensorRef<F32> {
    TensorRef {
        ptr: DevicePtr::new(1, len),
        dtype: DType::F32,
        layout,
    }
}

fn tmut_u32(len: usize, layout: Layout) -> TensorMut<U32> {
    TensorMut {
        ptr: DevicePtr::new(1, len),
        dtype: DType::U32,
        layout,
    }
}
