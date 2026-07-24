//! Coordinator for the checkpoint-native MTP draft lane.

use anyhow::Context as _;

use super::RankSlots;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::Glm52StepShape;
use crate::runner::Glm52MtpAppend;
use crate::runner::Glm52MtpRoundMode;
use crate::runner::Glm52Worker;

fn select_round_mode(
    rank_appends: &[Vec<Glm52MtpAppend>],
    rank_proposals: &[Vec<(usize, u32, usize)>],
) -> Glm52MtpRoundMode {
    if rank_proposals.iter().any(|proposals| !proposals.is_empty()) {
        Glm52MtpRoundMode::Propose
    } else if rank_appends.iter().any(|appends| !appends.is_empty()) {
        Glm52MtpRoundMode::ContextOnly
    } else {
        Glm52MtpRoundMode::ResetOnly
    }
}

/// Native MTP is an EP collective, unlike DSpark. Every worker receives every
/// round, including empty/padded ranks, and all use the same packed context
/// and proposal buckets for each of layer 78's five forwards.
pub(super) fn run_mtp_round(
    workers: &[Glm52Worker],
    slots: &mut [RankSlots],
    shapes: &[Glm52StepShape],
    pending_resets: &mut [Vec<usize>],
    rank_appends: Vec<Vec<Glm52MtpAppend>>,
    rank_proposals: Vec<Vec<(usize, u32, usize)>>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        workers.len() == slots.len()
            && shapes.len() == workers.len()
            && pending_resets.len() == workers.len()
            && rank_appends.len() == workers.len()
            && rank_proposals.len() == workers.len(),
        "GLM5.2 native MTP requires one logical rank per local EP worker"
    );
    let source_bucket = shapes
        .first()
        .context("GLM5.2 native MTP round has no target shape")?
        .bucket;
    anyhow::ensure!(
        shapes.iter().all(|shape| shape.bucket == source_bucket),
        "GLM5.2 native MTP source buckets diverged across EP ranks"
    );

    let pick_bucket = |rows: usize| {
        GLM52_DECODE_BUCKETS
            .into_iter()
            .find(|&bucket| bucket >= rows.max(1))
            .with_context(|| format!("GLM5.2 native MTP row count {rows} exceeds bucket capacity"))
    };
    let context_bucket = pick_bucket(rank_appends.iter().map(Vec::len).max().unwrap_or(0))?;
    let draft_bucket = pick_bucket(rank_proposals.iter().map(Vec::len).max().unwrap_or(0))?;
    let mode = select_round_mode(&rank_appends, &rank_proposals);

    let mut joins = Vec::with_capacity(workers.len());
    let mut proposal_slots = Vec::with_capacity(workers.len());
    for (rank, ((worker, appends), proposals)) in workers
        .iter()
        .zip(rank_appends)
        .zip(rank_proposals)
        .enumerate()
    {
        let slots_for_rank = proposals
            .iter()
            .map(|&(slot, _, _)| slot)
            .collect::<Vec<_>>();
        joins.push(worker.mtp_draft_async(
            mode,
            source_bucket,
            context_bucket,
            draft_bucket,
            std::mem::take(&mut pending_resets[rank]),
            appends,
            slots_for_rank.clone(),
        )?);
        proposal_slots.push(slots_for_rank);
    }

    for (rank, (rx, proposal_slots)) in joins.into_iter().zip(proposal_slots).enumerate() {
        let spans = rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its MTP response"))??;
        anyhow::ensure!(
            spans.len() == proposal_slots.len(),
            "GLM5.2 rank {rank} MTP returned {} spans for {} proposals",
            spans.len(),
            proposal_slots.len()
        );
        for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
            if let Some(active) = slots[rank][slot_id].as_mut() {
                active
                    .state
                    .set_drafts(span.to_vec(), crate::mtp::GLM52_MTP_DRAFTS);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn append() -> Glm52MtpAppend {
        Glm52MtpAppend {
            source_row: 0,
            slot: 0,
            input_token: 1,
            position: 0,
        }
    }

    #[test]
    fn any_rank_proposal_keeps_every_rank_in_the_collective_chain() {
        let appends = vec![vec![append()], Vec::new()];
        let proposals = vec![vec![(0, 1, 0)], Vec::new()];
        assert_eq!(
            select_round_mode(&appends, &proposals),
            Glm52MtpRoundMode::Propose
        );
    }

    #[test]
    fn committed_context_without_proposals_runs_only_the_first_pass() {
        let appends = vec![Vec::new(), vec![append()]];
        let proposals = vec![Vec::new(), Vec::new()];
        assert_eq!(
            select_round_mode(&appends, &proposals),
            Glm52MtpRoundMode::ContextOnly
        );
    }

    #[test]
    fn an_empty_round_only_resets_host_state() {
        let appends = vec![Vec::new(), Vec::new()];
        let proposals = vec![Vec::new(), Vec::new()];
        assert_eq!(
            select_round_mode(&appends, &proposals),
            Glm52MtpRoundMode::ResetOnly
        );
    }
}
