//! Offline replica of the coordinator's engine-fatal KV contract: the exact
//! schedule/apply sequence of the submit walk, driven end to end against a
//! real [`BlockPool`] — a schedule failure in serving tears the whole EP8
//! engine down, so the full-lifetime reservation must be proven tight here.

use openinfer_core::engine::FinishReason;
use openinfer_kv_cache::BlockPool;

use crate::model::GLM52_MAX_BATCH_PER_RANK;

use super::PAGE;
use super::admission::lifetime_blocks;
use super::slot::{Glm52SlotState, Glm52StepOutcome};
use super::testkit::EOS;

/// Drive one request end to end through the coordinator's exact
/// schedule/apply sequence against `pool` — the offline replica of the
/// two engine-fatal submit-walk assertions (span start == `kv_position`,
/// schedule never fails under the admission reservation). Verify spans
/// fully accept their drafts, maximizing the KV draw per round. Returns
/// the first schedule failure (the tight-budget control asserts one).
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
            state.set_drafts(vec![70_001, 70_002, 70_003, 70_004, 70_005, 70_006, 70_007]);
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
    // teardown; this is that contract's offline test. A pool sized
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
    state.set_drafts(vec![21, 7, 23]);
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
