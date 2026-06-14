use anyhow::Result;

use crate::speculative::{
    DraftPlan as SpeculativeDraftPlan, DraftResult as SpeculativeDraftResult,
    VerifyPlan as SpeculativeVerifyPlan, VerifyResult as SpeculativeVerifyResult,
};

use super::{Qwen3Executor, StepCommand, WorkerStepOutcome};

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
        for req in plan.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .expect("RequestKv was validated before speculative scheduling");
            rkv.schedule_speculative(req.token_ids.len(), self.kv_mgr.pool())
                .map_err(|e| {
                    anyhow::anyhow!("schedule_speculative failed for {:?}: {e}", req.request_id)
                })?;
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
        let outcome = self.run_step(&step)?;

        let result = match outcome {
            WorkerStepOutcome::SpeculativeVerify(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "speculative verify returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after speculative verify");
            rkv.apply_speculative(&req_result.accepted_tokens, self.kv_mgr.pool())?;
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
}
