use std::collections::HashMap;

use anyhow::{Context, Result};
use pegainfer_core::kv_pool::{KvExecView, KvPool, KvState};

use crate::request::RequestId;

pub(crate) struct KvBudgetState {
    pub(crate) current_tokens: usize,
    pub(crate) max_tokens: usize,
}

pub(crate) struct KvBudgetRequest {
    pub(crate) prompt_tokens: usize,
    pub(crate) max_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvAdmission {
    Admit,
    Defer,
    Reject,
}

pub(crate) struct KvExecViewBatch(Vec<KvExecView>);

impl KvExecViewBatch {
    pub(crate) fn views(&self) -> &[KvExecView] {
        &self.0
    }
}

pub(crate) struct UnifiedKvExecViews {
    pub(crate) prefill: KvExecViewBatch,
    pub(crate) decode: KvExecViewBatch,
}

impl UnifiedKvExecViews {
    fn new(prefill: KvExecViewBatch, decode: KvExecViewBatch) -> Self {
        Self { prefill, decode }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KvAppend {
    request_id: RequestId,
    append_tokens: usize,
    source: &'static str,
}

impl KvAppend {
    pub(crate) fn prefill(request_id: RequestId, append_tokens: usize) -> Self {
        Self {
            request_id,
            append_tokens,
            source: "prefill",
        }
    }

    pub(crate) fn decode(request_id: RequestId) -> Self {
        Self {
            request_id,
            append_tokens: 1,
            source: "decode",
        }
    }
}

fn append_request_ids(appends: &[KvAppend]) -> Vec<RequestId> {
    appends.iter().map(|append| append.request_id).collect()
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

struct PlannedGrow {
    request_id: RequestId,
    target_tokens: usize,
    source: &'static str,
}

struct RankPreparePlan {
    rank: usize,
    grows: Vec<PlannedGrow>,
}

struct RankKvStateStore {
    states: HashMap<RequestId, KvState>,
}

impl RankKvStateStore {
    fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    fn ensure_with<F>(&mut self, request_ids: &[RequestId], mut alloc: F)
    where
        F: FnMut() -> KvState,
    {
        for &request_id in request_ids {
            self.states.entry(request_id).or_insert_with(&mut alloc);
        }
    }

    fn drop_request(&mut self, request_id: RequestId) {
        self.states.remove(&request_id);
    }

    fn plan_appends(
        &self,
        rank: usize,
        pool: &KvPool,
        appends: &[KvAppend],
    ) -> Result<RankPreparePlan> {
        let page_size = pool.layout().page_size;
        let mut total_grow_pages = 0usize;
        let mut grows = Vec::new();

        for append in appends {
            let state = self
                .states
                .get(&append.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing {} KV state", append.source))?;
            let seq_len = state.seq_len();
            let held_pages = state.num_pages();
            let target_tokens = seq_len + append.append_tokens;
            let available_pages = pool.available_pages();
            let needed_pages = pages_needed(target_tokens, page_size);
            let grow_pages = needed_pages.saturating_sub(held_pages);
            total_grow_pages += grow_pages;

            if grow_pages > 0 {
                grows.push(PlannedGrow {
                    request_id: append.request_id,
                    target_tokens,
                    source: append.source,
                });
            }

            if total_grow_pages > available_pages {
                anyhow::bail!(
                    "rank {rank} prepare {} KV failed: request_id={}, seq_len={}, append_tokens={}, target_tokens={}, held_pages={}, grow_pages={}, total_grow_pages={}, available_pages={}",
                    append.source,
                    append.request_id.get(),
                    seq_len,
                    append.append_tokens,
                    target_tokens,
                    held_pages,
                    grow_pages,
                    total_grow_pages,
                    available_pages
                );
            }
        }

        Ok(RankPreparePlan { rank, grows })
    }

    fn apply_plan(&mut self, plan: RankPreparePlan) -> Result<()> {
        for grow in plan.grows {
            self.states
                .get_mut(&grow.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing {} KV state", grow.source))?
                .ensure_capacity(grow.target_tokens)
                .with_context(|| {
                    format!(
                        "rank {} apply {} KV grow failed: request_id={}, target_tokens={}",
                        plan.rank,
                        grow.source,
                        grow.request_id.get(),
                        grow.target_tokens
                    )
                })?;
        }
        Ok(())
    }

    fn build_views(
        &self,
        appends: &[KvAppend],
        missing_context: &'static str,
    ) -> Result<KvExecViewBatch> {
        Ok(KvExecViewBatch(
            appends
                .iter()
                .map(|append| {
                    self.states
                        .get(&append.request_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!("{missing_context} for {:?}", append.request_id)
                        })?
                        .exec_view(append.append_tokens)
                        .with_context(|| {
                            format!(
                                "build {} KV exec view failed: request_id={}, append_tokens={}",
                                append.source,
                                append.request_id.get(),
                                append.append_tokens
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

pub(crate) struct Qwen3KvCache {
    pools: Vec<KvPool>,
    rank_states: Vec<RankKvStateStore>,
}

impl Qwen3KvCache {
    pub(crate) fn new(pools: Vec<KvPool>) -> Self {
        let rank_states = pools.iter().map(|_| RankKvStateStore::new()).collect();
        Self { pools, rank_states }
    }

    fn max_request_physical_pages(&self) -> usize {
        self.pools
            .iter()
            .map(|pool| pool.capacity_pages().saturating_sub(1))
            .min()
            .unwrap_or(0)
    }

    fn available_physical_pages(&self) -> usize {
        self.pools
            .iter()
            .map(KvPool::available_pages)
            .min()
            .unwrap_or(0)
    }

    pub(crate) fn admit_requests(
        &self,
        active: &[KvBudgetState],
        pending: &[KvBudgetRequest],
    ) -> Vec<KvAdmission> {
        let page_size = self
            .pools
            .first()
            .map(|pool| pool.layout().page_size)
            .unwrap_or(1);
        let active_future_physical_pages: usize = active
            .iter()
            .map(|req| {
                pages_needed(req.max_tokens, page_size)
                    .saturating_sub(pages_needed(req.current_tokens, page_size))
            })
            .sum();
        let mut budget = self
            .available_physical_pages()
            .saturating_sub(active_future_physical_pages);
        let max_request_physical_pages = self.max_request_physical_pages();

        pending
            .iter()
            .map(|req| {
                debug_assert!(req.prompt_tokens <= req.max_tokens);
                let max_needed = pages_needed(req.max_tokens, page_size);
                if max_needed > max_request_physical_pages {
                    KvAdmission::Reject
                } else if max_needed <= budget {
                    budget -= max_needed;
                    KvAdmission::Admit
                } else {
                    KvAdmission::Defer
                }
            })
            .collect()
    }

    pub(crate) fn drop_request(&mut self, request_id: RequestId) {
        for store in &mut self.rank_states {
            store.drop_request(request_id);
        }
    }

    pub(crate) fn commit_prefill(&mut self, appends: &[KvAppend]) -> Result<()> {
        self.commit_appends(appends)
    }

    pub(crate) fn commit_decode(&mut self, appends: &[KvAppend]) -> Result<()> {
        self.commit_appends(appends)
    }

    pub(crate) fn commit_unified(
        &mut self,
        prefill_appends: &[KvAppend],
        decode_appends: &[KvAppend],
    ) -> Result<()> {
        let mut appends = Vec::with_capacity(prefill_appends.len() + decode_appends.len());
        appends.extend_from_slice(prefill_appends);
        appends.extend_from_slice(decode_appends);
        self.commit_appends(&appends)
    }

    pub(crate) fn prepare_prefill(&mut self, appends: &[KvAppend]) -> Result<Vec<KvExecViewBatch>> {
        let request_ids = append_request_ids(appends);
        for (rank, store) in self.rank_states.iter_mut().enumerate() {
            let pool = self.pools[rank].clone();
            store.ensure_with(&request_ids, || pool.alloc());
        }
        self.prepare_all_ranks(appends)?;
        self.build_rank_views(appends, "missing local prefill request state")
    }

    pub(crate) fn prepare_decode(&mut self, appends: &[KvAppend]) -> Result<Vec<KvExecViewBatch>> {
        self.prepare_all_ranks(appends)?;
        self.build_rank_views(appends, "missing local decode request state")
    }

    pub(crate) fn prepare_unified(
        &mut self,
        prefill_appends: &[KvAppend],
        decode_appends: &[KvAppend],
    ) -> Result<Vec<UnifiedKvExecViews>> {
        let mut all_appends = Vec::with_capacity(prefill_appends.len() + decode_appends.len());
        all_appends.extend_from_slice(prefill_appends);
        all_appends.extend_from_slice(decode_appends);
        let prefill_ids = append_request_ids(prefill_appends);

        for (rank, store) in self.rank_states.iter_mut().enumerate() {
            let pool = self.pools[rank].clone();
            store.ensure_with(&prefill_ids, || pool.alloc());
        }
        self.prepare_all_ranks(&all_appends)?;
        self.build_unified_rank_views(prefill_appends, decode_appends)
    }

    fn prepare_all_ranks(&mut self, appends: &[KvAppend]) -> Result<()> {
        let plans = self
            .rank_states
            .iter()
            .enumerate()
            .map(|(rank, store)| store.plan_appends(rank, &self.pools[rank], appends))
            .collect::<Result<Vec<_>>>()?;

        for (store, plan) in self.rank_states.iter_mut().zip(plans) {
            store.apply_plan(plan)?;
        }
        Ok(())
    }

    fn build_rank_views(
        &self,
        appends: &[KvAppend],
        missing_context: &'static str,
    ) -> Result<Vec<KvExecViewBatch>> {
        self.rank_states
            .iter()
            .map(|store| store.build_views(appends, missing_context))
            .collect()
    }

    fn build_unified_rank_views(
        &self,
        prefill_appends: &[KvAppend],
        decode_appends: &[KvAppend],
    ) -> Result<Vec<UnifiedKvExecViews>> {
        self.rank_states
            .iter()
            .map(|store| {
                let prefill = store.build_views(
                    prefill_appends,
                    "missing local unified prefill request state",
                )?;
                let decode = store
                    .build_views(decode_appends, "missing local unified decode request state")?;
                Ok(UnifiedKvExecViews::new(prefill, decode))
            })
            .collect()
    }

    fn commit_appends(&mut self, appends: &[KvAppend]) -> Result<()> {
        for (rank, store) in self.rank_states.iter().enumerate() {
            for append in appends {
                store.states.get(&append.request_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing {} KV state at commit for rank {}",
                        append.source,
                        rank
                    )
                })?;
            }
        }

        for store in &mut self.rank_states {
            for append in appends {
                store
                    .states
                    .get_mut(&append.request_id)
                    .expect("commit state presence checked above")
                    .advance(append.append_tokens);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pegainfer_core::tensor::DeviceContext;

    use super::*;

    fn test_pool(ctx: &DeviceContext, available_request_pages: usize) -> KvPool {
        KvPool::new(ctx, 1, 1, 1, 16, available_request_pages + 1).expect("test KvPool allocation")
    }

    #[test]
    fn prepare_prefill_does_not_partially_grow_physical_pools() {
        let ctx = DeviceContext::new().expect("GPU required for KV cache tests");
        let rank0_pool = test_pool(&ctx, 2);
        let rank1_pool = test_pool(&ctx, 1);
        let mut cache = Qwen3KvCache::new(vec![rank0_pool.clone(), rank1_pool.clone()]);

        let err = match cache.prepare_prefill(&[KvAppend::prefill(RequestId::new(7), 32)]) {
            Ok(_) => panic!("rank1 should not have enough physical KV pages"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("rank 1 prepare prefill KV failed"),
            "unexpected error: {err:#}"
        );
        assert_eq!(
            rank0_pool.available_pages(),
            2,
            "rank0 must not grow before rank1 preflight succeeds"
        );
        assert_eq!(
            rank1_pool.available_pages(),
            1,
            "failing rank must not change physical page ownership"
        );
    }
}
