use pegainfer_kv_cache::KvCacheManager;

fn make_manager(num_blocks: usize) -> KvCacheManager {
    let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    // Tiny geometry: 1 layer, 1 KV head, dim=1, block_size=16
    KvCacheManager::new(&stream, 1, 1, 1, 16, num_blocks).expect("KvCacheManager::new")
}

#[test]
fn single_request_prefill_decode_release() {
    let mgr = make_manager(32);
    // 1 block reserved for padding
    assert_eq!(mgr.total_blocks(), 32);
    let initial_avail = mgr.available_blocks();
    assert_eq!(initial_avail, 31); // 32 - 1 padding

    // New request: 10-token prompt, up to 6 output tokens
    let mut req = mgr.new_request(vec![1; 10], 6);
    assert_eq!(req.kv_position(), 0);

    // Schedule prefill for 10 tokens → needs 1 block (ceil(10/16))
    req.schedule_prefill(10, &mgr).expect("schedule_prefill");
    let view = req.prefill_view(10);
    assert_eq!(view.seq_len(), 10);
    assert_eq!(view.num_pages(), 1);
    assert_eq!(view.last_page_len(), 10);
    view.ensure_capacity(10).expect("capacity check");

    // Apply prefill (first generated token = 42)
    req.apply_prefill(42, &mgr).expect("apply_prefill");
    assert_eq!(req.kv_position(), 10);
    assert_eq!(req.generated_tokens(), 1);

    // Decode loop: 5 more tokens (total 6 output = max)
    for i in 0..5 {
        req.schedule_decode(&mgr).expect("schedule_decode");
        let view = req.decode_view();
        assert_eq!(view.seq_len(), 10 + 1 + i); // prompt + generated so far + 1 new
        req.apply_decode(100 + i as u32, &mgr)
            .expect("apply_decode");
    }

    assert_eq!(req.generated_tokens(), 6);
    assert!(req.is_complete());

    // Release → blocks return to pool
    req.release().expect("release");
    assert_eq!(mgr.available_blocks(), initial_avail);
}

#[test]
fn multiple_requests_share_capacity() {
    let mgr = make_manager(10); // 10 blocks, 1 padding → 9 usable
    assert_eq!(mgr.available_blocks(), 9);

    // Request A: 16 tokens prompt → 1 block, max 16 output → needs ceil(31/16)=2 blocks total
    let mut a = mgr.new_request(vec![1; 16], 16);
    a.schedule_prefill(16, &mgr).expect("a prefill schedule");
    a.apply_prefill(42, &mgr).expect("a prefill apply");

    let after_a = mgr.available_blocks();
    assert!(after_a < 9);

    // Request B: 16 tokens prompt → 1 block
    let mut b = mgr.new_request(vec![2; 16], 1);
    b.schedule_prefill(16, &mgr).expect("b prefill schedule");
    b.apply_prefill(43, &mgr).expect("b prefill apply");

    let after_b = mgr.available_blocks();
    assert!(after_b < after_a);

    // Release both → all blocks back
    a.release().expect("a release");
    b.release().expect("b release");
    assert_eq!(mgr.available_blocks(), 9);
}

#[test]
fn page_boundary_crossing() {
    let mgr = make_manager(10);

    // Prompt exactly 16 tokens (fills block 0 exactly), then decode
    // 1 token which should cross into block 1.
    let mut req = mgr.new_request(vec![1; 16], 2);
    req.schedule_prefill(16, &mgr).expect("schedule_prefill");

    let view = req.prefill_view(16);
    assert_eq!(view.num_pages(), 1);
    assert_eq!(view.last_page_len(), 16); // full page

    req.apply_prefill(42, &mgr).expect("apply_prefill");

    // First decode: token 42 is "dangling" (not yet in KV). schedule_decode
    // will allocate a new block for it.
    req.schedule_decode(&mgr).expect("schedule_decode");
    let view = req.decode_view();
    assert_eq!(view.seq_len(), 17); // 16 prompt + 1 decode
    assert_eq!(view.num_pages(), 2); // crossed page boundary
    assert_eq!(view.last_page_len(), 1); // 1 token in new page

    req.apply_decode(43, &mgr).expect("apply_decode");
    req.release().expect("release");
}

#[test]
fn kv_view_desc_has_correct_layout() {
    let mgr = make_manager(10);
    let mut req = mgr.new_request(vec![1; 5], 1);

    req.schedule_prefill(5, &mgr).expect("schedule_prefill");
    let view = req.prefill_view(5);
    let desc = view.desc(mgr.buffer());

    assert_eq!(desc.seq_len(), 5);
    assert_eq!(desc.last_page_len(), 5);
    assert_eq!(desc.num_pages(), 1);
    assert_eq!(desc.layout().page_size, 16);
    assert_eq!(desc.layout().num_layers, 1);
    assert!(!desc.buffer().is_empty());
    assert_eq!(desc.page_indices().len(), 1);

    req.apply_prefill(42, &mgr).expect("apply_prefill");
    req.release().expect("release");
}

#[test]
fn padding_block_id_is_stable() {
    let mgr = make_manager(10);
    let pid = mgr.padding_block_id();
    assert!(pid >= 0);
    // Allocate and release — padding ID must not change.
    let mut req = mgr.new_request(vec![1; 8], 1);
    req.schedule_prefill(8, &mgr).unwrap();
    req.apply_prefill(1, &mgr).unwrap();
    req.release().unwrap();
    assert_eq!(mgr.padding_block_id(), pid);
}

#[test]
fn exhaust_capacity_returns_error() {
    let mgr = make_manager(4); // 4 blocks, 1 padding → 3 usable

    // Each request with 16-token prompt needs 1 block.
    let mut reqs: Vec<_> = (0..3)
        .map(|_| {
            let mut r = mgr.new_request(vec![1; 16], 1);
            r.schedule_prefill(16, &mgr).expect("schedule");
            r.apply_prefill(42, &mgr).expect("apply");
            r
        })
        .collect();

    assert_eq!(mgr.available_blocks(), 0);

    // Next request should fail to schedule.
    let mut overflow = mgr.new_request(vec![1; 16], 1);
    let result = overflow.schedule_prefill(16, &mgr);
    assert!(result.is_err(), "should fail when no blocks available");

    for r in &mut reqs {
        r.release().unwrap();
    }
    assert_eq!(mgr.available_blocks(), 3);
}

/// Schedule allocates blocks, but if we never apply (e.g. forward
/// failed), dropping RequestKv must return those blocks to the pool.
#[test]
fn schedule_without_apply_returns_blocks_on_drop() {
    let mgr = make_manager(10); // 9 usable
    let before = mgr.available_blocks();

    {
        let mut req = mgr.new_request(vec![1; 32], 1);
        // 32 tokens → ceil(32/16) = 2 blocks scheduled
        req.schedule_prefill(32, &mgr).expect("schedule_prefill");
        assert!(mgr.available_blocks() < before);
        // No apply_prefill — simulate forward failure. Drop req here.
    }

    assert_eq!(mgr.available_blocks(), before, "blocks must return on drop");
}

/// ensure_capacity on a view should fail if the view doesn't have
/// enough pages for the requested token count.
#[test]
fn ensure_capacity_rejects_insufficient_pages() {
    let mgr = make_manager(10);
    let mut req = mgr.new_request(vec![1; 8], 1);

    req.schedule_prefill(8, &mgr).expect("schedule_prefill");
    let view = req.prefill_view(8);

    // View has 1 page (ceil(8/16)). Asking for 17 tokens needs 2 pages.
    let err = view.ensure_capacity(17);
    assert!(
        err.is_err(),
        "should reject: 17 tokens needs 2 pages but view has 1"
    );

    // 16 tokens still fits in 1 page.
    view.ensure_capacity(16).expect("16 tokens fits 1 page");

    req.apply_prefill(42, &mgr).unwrap();
    req.release().unwrap();
}

/// decode_view must produce seq_len == kv_position + 1.
#[test]
fn decode_view_seq_len_invariant() {
    let mgr = make_manager(10);
    let mut req = mgr.new_request(vec![1; 10], 4);

    req.schedule_prefill(10, &mgr).unwrap();
    req.apply_prefill(42, &mgr).unwrap();
    assert_eq!(req.kv_position(), 10);

    for i in 0..3 {
        req.schedule_decode(&mgr).unwrap();
        let view = req.decode_view();
        assert_eq!(
            view.seq_len(),
            req.kv_position() + 1,
            "decode_view seq_len must be kv_position + 1 (iteration {i})"
        );
        req.apply_decode(100 + i, &mgr).unwrap();
    }

    req.release().unwrap();
}

/// prefill_view seq_len == kv_position + prompt_len, page count covers
/// that seq_len.
#[test]
fn prefill_view_covers_target_seq_len() {
    let mgr = make_manager(10);
    let mut req = mgr.new_request(vec![1; 20], 1);

    req.schedule_prefill(20, &mgr).unwrap();
    let view = req.prefill_view(20);

    // seq_len = 0 (initial kv_position) + 20
    assert_eq!(view.seq_len(), 20);
    // 20 tokens → ceil(20/16) = 2 pages
    assert_eq!(view.num_pages(), 2);
    // ensure_capacity must agree
    view.ensure_capacity(20)
        .expect("20 tokens must fit in 2 pages");

    req.apply_prefill(42, &mgr).unwrap();
    req.release().unwrap();
}
