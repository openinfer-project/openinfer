//! Per-step planning: the global batch bucket and every rank's row list
//! ([`plan_step_shapes`]), the all-ranks-or-none launch-ahead decision
//! ([`launch_ahead_flags`]), and the rows the sampler owns instead of the
//! fused argmax ([`collect_sampling_rows`]) — pure functions over the same
//! fleet snapshot, so the collective-visible shapes can never disagree.

use openinfer_sample::SamplingParams;

use crate::config::GLM52_VOCAB;
use crate::model::{GLM52_DECODE_BUCKETS, GLM52_MAX_BATCH_PER_RANK, Glm52StepShape};
use crate::runner::{Glm52RowSample, Glm52StepFlags};

use super::slot::Glm52SlotState;
use super::{PAGE, RankSlots};

/// Every rank's forward shape for one step, decided together from the same
/// feed-want snapshot (`wants[rank][slot]` = rows that slot can usefully
/// fill: 0 free, 1 decode, remaining-prompt while mid-prefill).
///
/// The bucket is the smallest [`GLM52_DECODE_BUCKETS`] member covering the
/// hungriest rank's row demand (each rank's demand = Σ wants, capped at the
/// max bucket; never smaller than its active count — a smaller bucket would
/// silently drop rows). Per rank, every active slot first gets one row
/// (liveness), then the leftover bucket capacity extends mid-prefill slots
/// into *spans* (consecutive prompt positions batched through one step),
/// round-robin across the hungry slots so co-resident prefills drain in
/// parallel; padding rows ride the free slots. Span rows are emitted as one
/// contiguous run per slot — the [`Glm52StepShape`] contract.
/// Deriving the bucket and every rank's row list from the same data in one
/// place is what keeps them consistent.
/// `full_bucket` pins the bucket to `GLM52_MAX_BATCH_PER_RANK` regardless of
/// demand: the TP8 replicated topology serves exactly one graph shape (the
/// MoE phase kernels are fixed 8-row), so solo decode rides 7 padding rows
/// instead of a smaller bucket.
pub(super) fn plan_step_shapes(
    wants: &[[usize; GLM52_MAX_BATCH_PER_RANK]],
    full_bucket: bool,
) -> Vec<Glm52StepShape> {
    let hungriest = wants
        .iter()
        .map(|row| row.iter().sum::<usize>().min(GLM52_MAX_BATCH_PER_RANK))
        .max()
        .unwrap_or(0);
    let bucket = if full_bucket {
        GLM52_MAX_BATCH_PER_RANK
    } else {
        *GLM52_DECODE_BUCKETS
            .iter()
            .find(|&&rows| rows >= hungriest.max(1))
            .expect("the largest bucket covers every demand by construction")
    };
    wants
        .iter()
        .map(|row| {
            // Every active slot gets one row, then leftover capacity extends
            // spans one row per slot per round (round-robin), so two
            // mid-prefill slots on one rank drain in parallel instead of the
            // lowest slot starving the later one down to a liveness row for
            // its whole prefill.
            let mut spans = [0usize; GLM52_MAX_BATCH_PER_RANK];
            let mut used = 0usize;
            for (slot, &want) in row.iter().enumerate() {
                if want > 0 {
                    // bucket >= this rank's capped demand >= its active count
                    // by construction; a dropped active would stall forever.
                    assert!(used < bucket, "bucket {bucket} smaller than active count");
                    spans[slot] = 1;
                    used += 1;
                }
            }
            loop {
                let mut gave = false;
                for (slot, &want) in row.iter().enumerate() {
                    if used < bucket && spans[slot] > 0 && spans[slot] < want {
                        spans[slot] += 1;
                        used += 1;
                        gave = true;
                    }
                }
                if !gave || used == bucket {
                    break;
                }
            }
            let mut slots: [u8; GLM52_MAX_BATCH_PER_RANK] = std::array::from_fn(|slot| slot as u8);
            let mut dst = 0usize;
            for (slot, &span) in spans.iter().enumerate() {
                for _ in 0..span {
                    slots[dst] = slot as u8;
                    dst += 1;
                }
            }
            // Padding rows on free slots: there are always enough, because
            // used >= actives and bucket <= MAX, so bucket - used <= frees.
            let active_rows = dst;
            let mut frees = (0..GLM52_MAX_BATCH_PER_RANK).filter(|&slot| row[slot] == 0);
            while dst < bucket {
                slots[dst] = frees.next().expect("bucket - used <= free slots") as u8;
                dst += 1;
            }
            Glm52StepShape {
                bucket,
                slots,
                active_rows,
            }
        })
        .collect()
}

/// The launch-ahead flag decision — pure so the desync rules are testable.
/// `consume`: this step IS the speculation every rank enqueued (same shapes
/// AND no slot changed hands — a finish + admission can reuse a slot id
/// under an identical-looking shape). `lease`: every rank must enqueue the
/// next step speculatively — pure single-token GREEDY decode everywhere (the
/// speculation feeds each row's argmax token, so a sampled row would replay
/// the wrong input) with model-length headroom, off every 64-token page
/// boundary (the feed kernel's `slot_mapping += 1` only stays valid inside
/// the current page, and the advanced step's page must already be in the
/// uploaded block table; breaking the streak at every active row's boundary
/// also bounds padding rows — reset to position 0 by each full prologue —
/// inside the padding page), nothing queued, no draft round. Both are global
/// claims: a speculative replay is a full set of collectives, so per-rank
/// discretion would desync the pairing.
///
/// `offload_enabled` kills the lease outright: a leased replay keeps writing
/// KV on the rank stream for ~a step after the coordinator joined its
/// argmax D2H, and the offload restore leg H2Ds into freshly-reallocated
/// pool pages on pegaflow's OWN stream at the very next admission — the two
/// are unordered, so a replay row landing after the restore would silently
/// poison a content-addressed block for every later match. Without leases,
/// the joined D2H is the last thing on the rank stream and admission truly
/// is a quiet boundary. Costs ~0.7 ms/step, offload deployments only.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_ahead_flags(
    shapes: &[Glm52StepShape],
    leased_shapes: Option<&[Glm52StepShape]>,
    slots_changed: bool,
    pending_empty: bool,
    dspark_enabled: bool,
    offload_enabled: bool,
    slots: &[RankSlots],
    max_model_len: usize,
) -> Glm52StepFlags {
    let consume = !slots_changed && leased_shapes == Some(shapes);
    let lease = pending_empty
        && !dspark_enabled
        && !offload_enabled
        && slots
            .iter()
            .flat_map(|rank_slots| rank_slots.iter().flatten())
            .all(|active| {
                takes_argmax(&active.req.params) && lease_ok(&active.state, max_model_len)
            });
    Glm52StepFlags { consume, lease }
}

/// Whether a request's committed rows take the fused argmax — the shared
/// effectively-greedy predicate over the GLM vocab (a `top_p <= 1/vocab`
/// nucleus holds only the argmax token; routing it to the sampler would make
/// bf16-tied maxima stochastic, diverging from `select_batch`'s semantics).
/// The SAME predicate gates lease-granting and sampling-row collection, which
/// is what keeps "sampled row never rides a launch-ahead step" structural.
fn takes_argmax(params: &SamplingParams) -> bool {
    openinfer_sample::effectively_greedy(params, GLM52_VOCAB)
}

/// Whether one active request's KV position permits leasing the next step: a
/// pure single-token decode row with model-length headroom whose advanced
/// position stays inside its current 64-token page (see
/// [`launch_ahead_flags`] for why the page boundary breaks the streak).
fn lease_ok(state: &Glm52SlotState, max_model_len: usize) -> bool {
    let position = state.next_input_at(0).position;
    state.feed_want() == 1 && position + 1 < max_model_len && !(position + 1).is_multiple_of(PAGE)
}

/// The step rows a rank samples instead of argmaxes: walk the shape's
/// contiguous per-slot runs and mark each non-greedy slot's committable rows
/// (see [`Glm52SlotState::sampling_rows`]) with their request params and
/// request-local decode steps. Rows come out strictly ascending — the runs
/// are disjoint and walked in order, offsets ascend within a run — which
/// `sample_rows_into` re-checks.
pub(super) fn collect_sampling_rows(
    shape: &Glm52StepShape,
    rank_slots: &RankSlots,
) -> Vec<Glm52RowSample> {
    let mut sampling = Vec::new();
    let mut row = 0usize;
    while row < shape.bucket {
        let slot = shape.slots[row] as usize;
        let mut end = row + 1;
        while end < shape.bucket && shape.slots[end] as usize == slot {
            end += 1;
        }
        if let Some(active) = &rank_slots[slot]
            && !takes_argmax(&active.req.params)
        {
            for (offset, step) in active.state.sampling_rows(end - row) {
                sampling.push(Glm52RowSample {
                    row: row + offset,
                    params: active.req.params,
                    step,
                });
            }
        }
        row = end;
    }
    sampling
}

pub(super) fn feed_wants(slots: &[RankSlots]) -> Vec<[usize; GLM52_MAX_BATCH_PER_RANK]> {
    slots
        .iter()
        .map(|rank_slots| {
            std::array::from_fn(|slot| {
                rank_slots[slot]
                    .as_ref()
                    .map_or(0, |active| active.state.feed_want())
            })
        })
        .collect()
}

pub(super) fn occupancy(slots: &[RankSlots]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
    slots
        .iter()
        .map(|rank_slots| std::array::from_fn(|slot| rank_slots[slot].is_some()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::ActiveRequest;
    use crate::scheduler::slot::Glm52StepOutcome;
    use crate::scheduler::testkit::{EOS, commit, request, sampled, state, test_kv};

    fn shape(bucket: usize, active_rows: usize) -> Glm52StepShape {
        let mut slots = [0u8; GLM52_MAX_BATCH_PER_RANK];
        for (slot, dst) in slots.iter_mut().enumerate().take(bucket) {
            *dst = slot as u8;
        }
        Glm52StepShape {
            bucket,
            slots,
            active_rows,
        }
    }

    #[test]
    fn consume_requires_unchanged_shapes_and_untouched_slots() {
        let shapes = vec![shape(1, 1)];
        let flags =
            launch_ahead_flags(&shapes, Some(&shapes), false, true, false, false, &[], 4096);
        assert!(flags.consume);
    }

    #[test]
    fn slot_handoff_blocks_consume_even_under_identical_shapes() {
        // A finish + admission can reuse a slot id without changing the
        // shape — the desync class the first gate run hit.
        let shapes = vec![shape(1, 1)];
        let flags = launch_ahead_flags(&shapes, Some(&shapes), true, true, false, false, &[], 4096);
        assert!(!flags.consume);
    }

    #[test]
    fn active_row_count_is_part_of_shape_equality() {
        // Same bucket/slots but a row flipped active <-> pad must not consume:
        // a padding input is not value-distinguishable from an active one.
        let leased = vec![shape(1, 1)];
        let shapes = vec![shape(1, 0)];
        let flags =
            launch_ahead_flags(&shapes, Some(&leased), false, true, false, false, &[], 4096);
        assert!(!flags.consume);
    }

    #[test]
    fn no_lease_without_an_empty_queue() {
        let shapes = vec![shape(1, 1)];
        let flags = launch_ahead_flags(&shapes, None, false, false, false, false, &[], 4096);
        assert!(!flags.lease && !flags.consume);
    }

    /// One rank holding a single decoding request with the given params (its
    /// prompt token is already fed, so `feed_want() == 1`).
    fn decoding_fleet(params: openinfer_sample::SamplingParams) -> Vec<RankSlots> {
        let req = request(vec![10], params, 8);
        let mut state = Glm52SlotState::new(req.prompt_tokens.clone(), req.max_tokens, false, 0);
        assert!(matches!(
            state.advance_span(&[20], &[]),
            Glm52StepOutcome::Commit { .. }
        ));
        let kv = test_kv(req.prompt_tokens.clone(), req.max_tokens);
        let mut slots: RankSlots = std::array::from_fn(|_| None);
        slots[0] = Some(ActiveRequest { req, state, kv });
        vec![slots]
    }

    #[test]
    fn offload_blocks_the_lease() {
        // A leased replay keeps writing KV on the rank stream after the
        // join; the offload restore H2Ds on pegaflow's stream, unordered
        // against it. Offload on ⇒ never lease.
        let shapes = vec![shape(1, 1)];
        let greedy = decoding_fleet(openinfer_sample::SamplingParams::default());
        assert!(!launch_ahead_flags(&shapes, None, false, true, false, true, &greedy, 4096).lease);
    }

    #[test]
    fn non_greedy_request_blocks_the_lease() {
        // The speculation feeds each row's argmax token; a sampled row would
        // replay the wrong input, so any non-greedy active blocks the lease.
        let shapes = vec![shape(1, 1)];
        let greedy = decoding_fleet(openinfer_sample::SamplingParams::default());
        assert!(launch_ahead_flags(&shapes, None, false, true, false, false, &greedy, 4096).lease);

        let sampled = decoding_fleet(openinfer_sample::SamplingParams {
            temperature: 0.7,
            ..Default::default()
        });
        assert!(
            !launch_ahead_flags(&shapes, None, false, true, false, false, &sampled, 4096).lease
        );

        // An effectively-greedy request (top_p nucleus <= 1/vocab holds only
        // the argmax token) takes the argmax path, so it may ride the lease.
        let tiny_top_p = decoding_fleet(openinfer_sample::SamplingParams {
            temperature: 0.7,
            top_p: 0.5 / GLM52_VOCAB as f32,
            ..Default::default()
        });
        assert!(
            launch_ahead_flags(&shapes, None, false, true, false, false, &tiny_top_p, 4096).lease
        );
    }

    #[test]
    fn collect_sampling_rows_marks_each_spans_committable_rows() {
        // Bucket 8: slot 0 runs a 2-row verify span (non-greedy, drafts
        // installed), slot 1 finishes its prompt with a 3-row span
        // (non-greedy), slot 3 is mid-prompt (non-greedy, span does NOT
        // complete), slot 2 decodes greedily, row 7 pads.
        let shape = Glm52StepShape {
            bucket: 8,
            slots: [0, 0, 1, 1, 1, 3, 2, 4],
            active_rows: 7,
        };
        let mut rank_slots: RankSlots = std::array::from_fn(|_| None);

        let mut decode_state = state(vec![10], 8, false);
        assert_eq!(
            decode_state.advance_span(&[20], EOS),
            commit(&[20], 1, None, 1)
        );
        decode_state.set_drafts(
            vec![50, 51, 52],
            crate::scheduler::slot::GLM52_DSPARK_EP8_SPAN_DRAFTS,
        );
        rank_slots[0] = Some(ActiveRequest {
            req: request(vec![10], sampled(0.8), 8),
            state: decode_state,
            kv: test_kv(vec![10], 8),
        });

        let mut boundary_state = state(vec![10, 11, 12, 13, 14], 8, false);
        assert_eq!(
            boundary_state.advance_span(&[99, 98], EOS),
            Glm52StepOutcome::Prefilling
        );
        rank_slots[1] = Some(ActiveRequest {
            req: request(vec![10, 11, 12, 13, 14], sampled(0.8), 8),
            state: boundary_state,
            kv: test_kv(vec![10, 11, 12, 13, 14], 8),
        });

        let mut greedy_state = state(vec![10], 8, false);
        assert_eq!(
            greedy_state.advance_span(&[20], EOS),
            commit(&[20], 1, None, 1)
        );
        rank_slots[2] = Some(ActiveRequest {
            req: request(vec![10], openinfer_sample::SamplingParams::default(), 8),
            state: greedy_state,
            kv: test_kv(vec![10], 8),
        });

        rank_slots[3] = Some(ActiveRequest {
            req: request(vec![30; 10], sampled(0.8), 8),
            state: state(vec![30; 10], 8, false),
            kv: test_kv(vec![30; 10], 8),
        });

        let rows = collect_sampling_rows(&shape, &rank_slots);
        let picked: Vec<(usize, u64)> = rows.iter().map(|s| (s.row, s.step)).collect();
        // Slot 0's verify span samples BOTH rows (anchor row 0 at step 1,
        // draft row 1 at step 2 — the planner granted 2 of its 4 wanted
        // rows); slot 1's boundary span commits its LAST row (row 2 +
        // offset 2 = 4, first generated token → step 0). Slot 3's
        // mid-prompt span and slot 2's greedy row contribute nothing.
        assert_eq!(picked, vec![(0, 1), (1, 2), (4, 0)]);
    }

    #[test]
    fn effectively_greedy_rows_take_the_argmax_path() {
        // temperature > 0 but the top_p nucleus (<= 1/vocab) holds only the
        // argmax token: the row must NOT be collected for the sampler — the
        // FlashInfer pass could pick a different bf16-tied maximum, whereas
        // `select_batch` pins this case to the deterministic argmax.
        let shape = Glm52StepShape {
            bucket: 1,
            slots: [0; GLM52_MAX_BATCH_PER_RANK],
            active_rows: 1,
        };
        let mut rank_slots: RankSlots = std::array::from_fn(|_| None);
        let mut state = state(vec![10], 8, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        rank_slots[0] = Some(ActiveRequest {
            req: request(
                vec![10],
                openinfer_sample::SamplingParams {
                    top_p: 0.5 / GLM52_VOCAB as f32,
                    ..sampled(0.8)
                },
                8,
            ),
            state,
            kv: test_kv(vec![10], 8),
        });
        assert!(collect_sampling_rows(&shape, &rank_slots).is_empty());
    }

    #[test]
    fn lease_breaks_at_the_page_boundary() {
        // Anchor at position 62 → the next position 63 stays in page 0:
        // lease ok. Anchor at position 63 → position 64 opens page 1: the
        // feed kernel's `slot_mapping += 1` would leave the page — no lease.
        let mut s = state((0..63).collect(), 8, false);
        let mut outputs = vec![99u32; 63];
        *outputs.last_mut().unwrap() = 42;
        assert_eq!(s.advance_span(&outputs, EOS), commit(&[42], 1, None, 63));
        assert_eq!(s.next_input_at(0).position, 63);
        assert!(!lease_ok(&s, 4096), "position 63 -> 64 crosses the page");
        assert_eq!(s.advance_span(&[43], EOS), commit(&[43], 1, None, 1));
        assert_eq!(s.next_input_at(0).position, 64);
        assert!(lease_ok(&s, 4096), "position 64 -> 65 stays inside page 1");
        // Model-length headroom still gates.
        assert!(!lease_ok(&s, 65));
    }

    /// `counts` decode-phase requests per rank (each wants one row).
    fn decode_wants(counts: &[usize]) -> Vec<[usize; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| usize::from(slot < c)))
            .collect()
    }

    /// The observable part of a shape: the bucket and the forwarded rows'
    /// slots (trailing entries beyond the bucket are never read).
    fn forwarded(shapes: &[Glm52StepShape]) -> Vec<(usize, Vec<u8>)> {
        shapes
            .iter()
            .map(|shape| (shape.bucket, shape.slots[..shape.bucket].to_vec()))
            .collect()
    }

    #[test]
    fn full_bucket_pins_every_shape_to_the_max() {
        // TP8 replicated: one logical rank, one graph shape. A mid-prefill
        // request wanting 5 rows takes 5 rows + 3 pads.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 5;
        let shapes = plan_step_shapes(&wants, true);
        assert_eq!(forwarded(&shapes), vec![(8, vec![0, 0, 0, 0, 0, 1, 2, 3])]);
        assert_eq!(shapes[0].active_rows, 5);

        // Solo single-token decode still rides the full bucket (7 pads) —
        // the MoE phase kernels are fixed 8-row.
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[1]), true)),
            vec![(8, vec![0, 1, 2, 3, 4, 5, 6, 7])]
        );

        // Concurrency packs actives first, then pads.
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[3]), true)),
            vec![(8, vec![0, 1, 2, 3, 4, 5, 6, 7])]
        );

        // A verify span (slot wants 8) owns the whole bucket.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 8;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants, true)),
            vec![(8, vec![0; 8])]
        );
    }

    #[test]
    fn bucket_is_the_smallest_covering_the_hungriest_rank() {
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[0, 0]), false)),
            vec![(1, vec![0]), (1, vec![0])]
        );
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[1; 8]), false)),
            vec![(1, vec![0]); 8]
        );
        // One rank at two requests lifts EVERY rank to the 2-row bucket —
        // idle ranks pad with free slots.
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[2, 1]), false)),
            vec![(2, vec![0, 1]), (2, vec![0, 1])]
        );
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[3, 1]), false))[0],
            (4, vec![0, 1, 2, 3])
        );
        // Past the 4-row bucket the full batch takes over.
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[5, 1]), false))[0].0,
            8
        );
    }

    #[test]
    fn partial_buckets_pack_actives_first() {
        // A rank holding slots {1, 5} forwards them in rows 0..2; the padding
        // rows (bucket 4) ride on the lowest free slots.
        let mut holey = decode_wants(&[0, 3]);
        holey[0][1] = 1;
        holey[0][5] = 1;
        assert_eq!(
            forwarded(&plan_step_shapes(&holey, false)),
            vec![(4, vec![1, 5, 0, 2]), (4, vec![0, 1, 2, 3])]
        );
        let mut deep = decode_wants(&[5, 0]);
        deep[0][0] = 0;
        deep[0][7] = 1;
        assert_eq!(
            forwarded(&plan_step_shapes(&deep, false))[0],
            (8, vec![1, 2, 3, 4, 7, 0, 5, 6])
        );
    }

    #[test]
    fn prefill_want_extends_one_slot_into_a_span() {
        // A lone mid-prefill request with plenty of prompt left fills the
        // whole max bucket with its span; idle ranks pad.
        let mut wants = decode_wants(&[0, 0]);
        wants[0][2] = 3000;
        let shapes = plan_step_shapes(&wants, false);
        assert_eq!(
            forwarded(&shapes)[0],
            (8, vec![2, 2, 2, 2, 2, 2, 2, 2]),
            "one hungry slot owns every row of the max bucket"
        );
        assert_eq!(
            forwarded(&shapes)[1],
            (8, (0..8).map(|s| s as u8).collect())
        );

        // A short prompt remainder only lifts the bucket as far as needed.
        let mut wants = decode_wants(&[0, 0]);
        wants[0][0] = 3;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants, false))[0],
            (4, vec![0, 0, 0, 1])
        );
    }

    #[test]
    fn spans_share_the_bucket_with_decode_slots_actives_first() {
        // Slot 0 decodes (1 row), slot 1 is mid-prefill: liveness rows first,
        // then the leftover capacity extends the prefill span — one
        // contiguous run per slot.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 1;
        wants[0][1] = 100;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants, false))[0],
            (8, vec![0, 1, 1, 1, 1, 1, 1, 1])
        );

        // Two mid-prefill slots with small wants: both met, remaining rows
        // pad on free slots.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 3;
        wants[0][1] = 2;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants, false))[0],
            (8, vec![0, 0, 0, 1, 1, 2, 3, 4]),
            "wants met, remaining rows pad on free slots"
        );
    }

    #[test]
    fn two_long_prefills_split_the_leftover_round_robin() {
        // Two co-resident long prefills split the bucket evenly — neither
        // starves at a single liveness row while the other eats the leftover.
        let mut wants = decode_wants(&[0]);
        wants[0][2] = 3000;
        wants[0][5] = 3000;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants, false))[0],
            (8, vec![2, 2, 2, 2, 5, 5, 5, 5])
        );

        // A decode slot in the mix keeps its single row; the prefills split
        // what remains (7 rows -> 4 + 3 by round-robin order).
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 1;
        wants[0][3] = 3000;
        wants[0][6] = 3000;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants, false))[0],
            (8, vec![0, 3, 3, 3, 3, 6, 6, 6])
        );
    }
}
