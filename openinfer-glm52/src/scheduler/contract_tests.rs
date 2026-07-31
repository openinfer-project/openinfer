//! Offline replica of the engine's engine-fatal KV contract: the exact
//! schedule/apply sequence of the submit walk, driven end to end against a
//! real [`BlockPool`] — a schedule failure in serving is engine-fatal, so
//! the full-lifetime reservation must be proven tight here.

use std::collections::VecDeque;

use openinfer_core::engine::FinishReason;
use openinfer_core::engine::LoadSnapshot;
use openinfer_core::engine::TokenSink;
use openinfer_kv_cache::BlockPool;
use openinfer_sample::SamplingParams;

use super::ActiveRequest;
use super::PAGE;
use super::RankSlots;
use super::admission::lifetime_blocks;
use super::admit_from_queue;
use super::graph::graph_dump_bucket;
use super::publish_load;
use super::slot::GLM52_DSPARK_EP8_SPAN_DRAFTS;
use super::slot::Glm52SlotState;
use super::slot::Glm52StepOutcome;
use super::testkit::EOS;
use super::testkit::request;
use crate::model::GLM52_MAX_BATCH_PER_RANK;

#[test]
fn graph_dump_uses_bucket_one() {
    assert_eq!(graph_dump_bucket(), 1, "EP and TP4 export bucket-1 graphs");
}

#[test]
fn load_snapshot_reports_the_ranks_own_state() {
    let pool = BlockPool::new(PAGE, 8).expect("pool");
    let mut slots: RankSlots = std::array::from_fn(|_| None);

    let req = request(vec![10, 11], SamplingParams::default(), 4);
    let state = Glm52SlotState::new(req.prompt_tokens.clone(), req.max_tokens, true, 0);
    let mut kv = pool.new_request(req.prompt_tokens.clone(), req.max_tokens, None);
    kv.schedule_prefill(1, &pool).expect("one live KV block");
    slots[0] = Some(ActiveRequest {
        req,
        state,
        client_prompt_tokens: 2,
        kv,
    });

    let mut pending = VecDeque::new();
    pending.push_back(request(vec![20], SamplingParams::default(), 4));
    pending.push_back(request(vec![21], SamplingParams::default(), 4));

    let (load_tx, load_rx) = tokio::sync::watch::channel(LoadSnapshot::default());
    publish_load(&load_tx, &pool, &slots, &pending);

    let snapshot = *load_rx.borrow();
    assert_eq!(snapshot.num_running_reqs, 1);
    assert_eq!(snapshot.num_waiting_reqs, 2);
    assert_eq!(snapshot.kv_total_blocks, 7);
    assert_eq!(snapshot.kv_used_blocks, 1);
}

#[test]
fn admission_fills_free_slots_from_the_local_queue() {
    let pool = BlockPool::new(PAGE, 8).expect("pool");
    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut req = request(vec![10], SamplingParams::default(), 4);
    req.data_parallel_rank = Some(0);
    let (token_tx, _token_rx) = TokenSink::standalone();
    req.token_tx = token_tx;
    pending.push_back(req);
    let mut pending_resets = Vec::new();

    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        7,
        None,
        &mut None,
        &mut None,
        false,
        false,
        false,
        &mut pending_resets,
    )
    .expect("admission");

    assert!(slots[0].is_some());
    assert!(pending.is_empty());
}

#[test]
fn prefill_only_admits_multiple_requests_within_pool_capacity() {
    let pool = BlockPool::new(PAGE, 16).expect("pool");
    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut token_receivers = Vec::new();
    for token in [10, 20] {
        let mut req = request(vec![token], SamplingParams::default(), 1);
        let (token_tx, token_rx) = TokenSink::standalone();
        req.token_tx = token_tx;
        token_receivers.push(token_rx);
        pending.push_back(req);
    }
    let mut pending_resets = Vec::new();

    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        15,
        None,
        &mut None,
        &mut None,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("prefill-only admission");

    assert_eq!(slots.iter().flatten().count(), 2);
    assert!(pending.is_empty());
}

#[test]
fn admission_defers_while_physical_pages_are_temporarily_held() {
    let pool = BlockPool::new(PAGE, 6).expect("pool");
    let mut held = pool.new_request(vec![1; 2 * PAGE], 1, None);
    held.schedule_prefill(2 * PAGE, &pool)
        .expect("temporarily hold two physical pages");

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut req = request(vec![2; 3 * PAGE], SamplingParams::default(), 1);
    let (token_tx, _token_rx) = TokenSink::standalone();
    req.token_tx = token_tx;
    pending.push_back(req);
    let mut pending_resets = Vec::new();

    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        5,
        None,
        &mut None,
        &mut None,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("temporary pressure is not an admission error");

    assert_eq!(pending.len(), 1, "request stays queued");
    assert!(slots.iter().all(Option::is_none));

    held.revert_schedule().expect("release temporary pages");
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        5,
        None,
        &mut None,
        &mut None,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("admit after pressure clears");

    assert!(pending.is_empty());
    assert_eq!(slots.iter().flatten().count(), 1);
}

/// Drive one request end to end through the engine's exact schedule/apply
/// sequence against `pool` — the offline replica of the two engine-fatal
/// submit-walk assertions (span start == `kv_position`, schedule never fails
/// under the admission reservation). Verify spans fully accept their drafts,
/// maximizing the KV draw per round. Returns the first schedule failure (the
/// tight-budget control asserts one).
fn drive_request(
    pool: &BlockPool,
    prompt_len: usize,
    max_tokens: usize,
    with_drafts: bool,
) -> Result<(), String> {
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|t| 10_000 + t).collect();
    let mut state = Glm52SlotState::new(prompt.clone(), max_tokens, true, 0);
    let mut kv = pool.new_request(prompt, max_tokens, None);
    let mut fresh = 60_000u32;
    loop {
        if with_drafts && state.wants_drafts() {
            state.set_drafts(
                vec![70_001, 70_002, 70_003, 70_004, 70_005, 70_006, 70_007],
                GLM52_DSPARK_EP8_SPAN_DRAFTS,
            );
        }
        let span = state.feed_want().min(GLM52_MAX_BATCH_PER_RANK);
        assert_eq!(
            state.next_input_at(0).position,
            kv.kv_position(),
            "span start drifted from the pool's kv_position"
        );
        let mid_prefill = state.mid_prefill();
        if mid_prefill {
            kv.schedule_prefill(span, pool)
                .map_err(|e| format!("schedule_prefill: {e}"))?;
        } else if span == 1 {
            kv.schedule_decode(pool)
                .map_err(|e| format!("schedule_decode: {e}"))?;
        } else {
            kv.schedule_speculative(span, pool)
                .map_err(|e| format!("schedule_speculative: {e}"))?;
        }
        // The prologue's page-row coverage, offline: the exact page row
        // must cover every fed position.
        let pages = kv.step_page_indices(span);
        let last_position = state.next_input_at(span - 1).position;
        assert!(
            pages.len() * PAGE > last_position,
            "page row misses a fed position"
        );
        fresh += 1;
        // Rows 1.. echo the fed tokens (a verify span fully accepts its
        // drafts), the last row emits a fresh token.
        let outputs: Vec<u32> = (1..span)
            .map(|offset| state.next_input_at(offset).token)
            .chain(std::iter::once(fresh))
            .collect();
        match state.advance_span(&outputs, &[]) {
            Glm52StepOutcome::Prefilling => {
                kv.apply_prefill_chunk(pool).expect("apply_prefill_chunk");
            }
            Glm52StepOutcome::Commit {
                committed, finish, ..
            } => {
                if mid_prefill {
                    kv.apply_prefill(committed[0], pool).expect("apply_prefill");
                } else if span == 1 {
                    kv.apply_decode(committed[0], pool).expect("apply_decode");
                } else {
                    kv.apply_speculative(&committed, pool)
                        .expect("apply_speculative");
                }
                if finish.is_some() {
                    break;
                }
            }
        }
    }
    kv.release().map_err(|e| format!("release: {e}"))?;
    Ok(())
}

#[test]
fn full_lifetime_reservation_covers_kvbm_peak_draw() {
    // The submit walk turns any schedule failure into an engine
    // exit; this is that contract's offline test. A pool sized
    // exactly `lifetime_blocks + 1` (padding) must carry every shape end
    // to end — and one block less must NOT, or the reservation is merely
    // sufficient by accident, not tight.
    for &(prompt_len, max_tokens) in &[
        (64usize, 64usize),
        (64, 65),
        (63, 65),
        (1, 128),
        (127, 2),
        (192, 3),
        (65, 1),
    ] {
        for with_drafts in [false, true] {
            let lifetime = lifetime_blocks(prompt_len, max_tokens);
            let pool = BlockPool::new(PAGE, lifetime + 1).expect("pool");
            drive_request(&pool, prompt_len, max_tokens, with_drafts).unwrap_or_else(|e| {
                panic!("({prompt_len},{max_tokens},drafts={with_drafts}): {e}")
            });
            let tight = BlockPool::new(PAGE, lifetime).expect("tight pool");
            assert!(
                drive_request(&tight, prompt_len, max_tokens, with_drafts).is_err(),
                "({prompt_len},{max_tokens},drafts={with_drafts}): a budget below the \
                 lifetime must fail somewhere"
            );
        }
    }
}

#[test]
fn eos_truncated_speculative_apply_stays_in_contract() {
    // EOS mid-verify-span truncates `committed` (the suppressed EOS is
    // its last entry); `apply_speculative` with the truncated run and
    // the release must both stay clean.
    let pool = BlockPool::new(PAGE, 16).expect("pool");
    let prompt: Vec<u32> = (0..70).collect();
    let mut state = Glm52SlotState::new(prompt.clone(), 32, false, 0);
    let mut kv = pool.new_request(prompt, 32, None);
    loop {
        if !state.mid_prefill() {
            break;
        }
        let span = state.feed_want().min(GLM52_MAX_BATCH_PER_RANK);
        assert_eq!(state.next_input_at(0).position, kv.kv_position());
        kv.schedule_prefill(span, &pool).expect("schedule_prefill");
        match state.advance_span(&vec![50u32; span], EOS) {
            Glm52StepOutcome::Prefilling => {
                kv.apply_prefill_chunk(&pool).expect("apply_prefill_chunk");
            }
            Glm52StepOutcome::Commit { committed, .. } => {
                kv.apply_prefill(committed[0], &pool)
                    .expect("apply_prefill");
            }
        }
    }
    state.set_drafts(vec![21, 7, 23], GLM52_DSPARK_EP8_SPAN_DRAFTS);
    let span = state.feed_want();
    assert_eq!(span, 4, "anchor + 3 drafts");
    assert_eq!(state.next_input_at(0).position, kv.kv_position());
    kv.schedule_speculative(span, &pool)
        .expect("schedule_speculative");
    let outcome = state.advance_span(&[21, 7, 23, 99], EOS);
    let Glm52StepOutcome::Commit {
        committed,
        emit,
        finish,
        ..
    } = outcome
    else {
        panic!("verify span must commit");
    };
    assert_eq!(committed, vec![21, 7], "truncated to the consumed run");
    assert_eq!(emit, 1, "the suppressed EOS is consumed, not emitted");
    assert_eq!(finish, Some(FinishReason::Stop));
    kv.apply_speculative(&committed, &pool)
        .expect("apply_speculative with the truncated run");
    kv.release().expect("release");
}
