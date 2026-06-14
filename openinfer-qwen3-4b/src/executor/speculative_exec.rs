use anyhow::Result;

use crate::speculative::{
    DraftPlan as SpeculativeDraftPlan, DraftResult as SpeculativeDraftResult,
    VerifyPlan as SpeculativeVerifyPlan, VerifyResult as SpeculativeVerifyResult,
};

use super::{Qwen3Executor, RequestId, StepCommand, WorkerStepOutcome};

impl Qwen3Executor {
    pub(super) fn execute_speculative_verify_impl(
        &mut self,
        plan: SpeculativeVerifyPlan<'_>,
    ) -> Result<SpeculativeVerifyResult> {
        anyhow::ensure!(
            self.speculative_enabled,
            "speculative verification requested but no draft model is loaded"
        );
        for req in plan.requests {
            anyhow::ensure!(
                !req.token_ids.is_empty(),
                "speculative verify request {:?} has an empty verify span",
                req.request_id
            );
            anyhow::ensure!(
                req.params.is_greedy(),
                "speculative verification currently supports greedy sampling only"
            );
            anyhow::ensure!(
                self.dflash_ready_requests.contains(&req.request_id),
                "speculative verification requested before DFlash state is ready for {:?}",
                req.request_id
            );
            anyhow::ensure!(
                self.request_kvs.contains_key(&req.request_id),
                "missing RequestKv for {:?}",
                req.request_id
            );
        }

        let mut scheduled = Vec::with_capacity(plan.requests.len());
        for req in plan.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .expect("RequestKv was validated before speculative scheduling");
            if let Err(e) = rkv.schedule_speculative(req.token_ids.len(), self.kv_mgr.pool()) {
                self.revert_speculative_schedules(&scheduled);
                return Err(anyhow::anyhow!(
                    "schedule_speculative failed for {:?}: {e}",
                    req.request_id
                ));
            }
            scheduled.push(req.request_id);
        }

        let kv_views = plan
            .requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].speculative_view(req.token_ids.len()))
            .collect();

        let step = StepCommand::SpeculativeVerify {
            requests: plan.requests.to_vec(),
            kv_views,
        };
        let outcome = match self.run_step(&step) {
            Ok(outcome) => outcome,
            Err(e) => {
                self.revert_speculative_schedules(&scheduled);
                return Err(e);
            }
        };

        let result = match outcome {
            WorkerStepOutcome::SpeculativeVerify(result) => result,
            other => {
                self.revert_speculative_schedules(&scheduled);
                return Err(anyhow::anyhow!(
                    "speculative verify returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        if result.requests.len() != plan.requests.len() {
            self.revert_speculative_schedules(&scheduled);
            return Err(anyhow::anyhow!(
                "speculative verify returned {} request results for {} requests",
                result.requests.len(),
                plan.requests.len()
            ));
        }
        for (req, req_result) in plan.requests.iter().zip(&result.requests) {
            if req.request_id != req_result.request_id {
                self.revert_speculative_schedules(&scheduled);
                return Err(anyhow::anyhow!(
                    "speculative verify returned request {:?} for {:?}",
                    req_result.request_id,
                    req.request_id
                ));
            }
        }

        let mut applied = Vec::with_capacity(result.requests.len());
        for req_result in &result.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after speculative verify");
            if let Err(e) = rkv.apply_speculative(&req_result.accepted_tokens, self.kv_mgr.pool()) {
                let unapplied = scheduled
                    .iter()
                    .copied()
                    .filter(|request_id| !applied.contains(request_id))
                    .collect::<Vec<_>>();
                self.revert_speculative_schedules(&unapplied);
                return Err(anyhow::anyhow!(
                    "apply_speculative failed for {:?}: {e}",
                    req_result.request_id
                ));
            }
            applied.push(req_result.request_id);
        }
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    pub(super) fn execute_speculative_draft_impl(
        &mut self,
        plan: SpeculativeDraftPlan<'_>,
    ) -> Result<SpeculativeDraftResult> {
        anyhow::ensure!(
            self.speculative_enabled,
            "speculative draft requested but no draft model is loaded"
        );
        for req in plan.requests {
            anyhow::ensure!(
                req.params.is_greedy(),
                "speculative draft currently supports greedy sampling only"
            );
            anyhow::ensure!(
                self.dflash_ready_requests.contains(&req.request_id),
                "speculative draft requested before DFlash state is ready for {:?}",
                req.request_id
            );
        }
        let step = StepCommand::SpeculativeDraft {
            requests: plan.requests.to_vec(),
        };
        let outcome = self.run_step(&step)?;
        match outcome {
            WorkerStepOutcome::SpeculativeDraft(result) => Ok(result),
            other => Err(anyhow::anyhow!(
                "speculative draft returned unexpected: {}",
                other.kind()
            )),
        }
    }

    fn revert_speculative_schedules(&mut self, request_ids: &[RequestId]) {
        for request_id in request_ids.iter().rev().copied() {
            let Some(rkv) = self.request_kvs.get_mut(&request_id) else {
                log::warn!(
                    "missing RequestKv while reverting speculative schedule for {request_id:?}"
                );
                continue;
            };
            if let Err(error) = rkv.revert_schedule() {
                log::warn!("failed to revert speculative schedule for {request_id:?}: {error}");
            }
        }
    }
}
