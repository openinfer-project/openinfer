use std::collections::HashSet;

use anyhow::Result;

use crate::batch_decode_buffers::BATCH_BUCKETS;
use crate::config::TensorParallelConfig;
use openinfer_core::engine::{LoadLoraAdapterRequest, UnloadLoraAdapterRequest};
use openinfer_kv_cache::KvView;

use super::dflash_prefill::{DFlashPrefillAction, dflash_prefill_action};
use super::worker::{StepCommand, WorkerStepOutcome};
use super::{
    DecodePlan, DecodeResult, ModelExecutor, PrefillPlan, PrefillResult, Qwen3Executor, RequestId,
    SpeculativeDraftPlan, SpeculativeDraftResult, SpeculativeVerifyPlan, SpeculativeVerifyResult,
    UnifiedPlan, UnifiedResult,
};

impl ModelExecutor for Qwen3Executor {
    fn block_size(&self) -> usize {
        self.metadata.block_size
    }

    fn max_request_blocks(&self) -> usize {
        self.kv_mgr.pool().max_request_blocks()
    }

    fn max_context_tokens(&self) -> usize {
        self.metadata.max_context_tokens
    }

    fn max_decode_batch_size(&self) -> usize {
        *BATCH_BUCKETS.last().unwrap()
    }

    fn available_blocks(&self) -> usize {
        self.kv_mgr.pool().available_blocks()
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.metadata.stop_token_ids.contains(&token_id)
    }

    fn prefetched_blocks(&self, request_id: RequestId) -> usize {
        self.prefetch
            .get(&request_id)
            .map_or(0, |st| st.probe.held_blocks())
    }

    fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        // Remove and drop — RAII on SchedulableSequence's block guards
        // returns all allocated blocks regardless of lifecycle state. The same
        // RAII frees any parked prefetch's reserved/held blocks.
        self.request_kvs.remove(&request_id);
        // A parked prefetch may still have a load in flight: pegaflow's worker
        // is writing the reserved GPU blocks (H2D). Dropping the reservation now
        // frees those physical pages for immediate reuse while the DMA keeps
        // landing on them — silent KV corruption, the load-side mirror of the
        // SAVE keep-alive pin. Block until the copy finishes before the
        // reservation drops. The scheduler is a dedicated synchronous thread, so
        // this brief wait costs nothing it could spend elsewhere.
        if let Some(mut state) = self.prefetch.remove(&request_id) {
            if let Some(handle) = state.handle.take() {
                let _ = handle.wait();
            }
        }
        self.saved_cursor.remove(&request_id);
        if self.speculative_enabled {
            self.dflash_ready_requests.remove(&request_id);
            self.primary.drop_dflash_request(request_id)?;
        }
        Ok(())
    }

    fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        lora_adapter: Option<&str>,
        reserve_floor: usize,
    ) -> bool {
        let Some(offload) = self.offload.as_ref() else {
            return false;
        };
        if !self.prefix_cache_enabled {
            return false;
        }
        if self.l1_retention_disabled {
            // Pure-L2 mode: drop any cross-request HBM retention so the probe
            // sees gpu_hit == 0 and queries the whole cacheable prefix from the
            // host tier. Only inactive (completed, unheld) blocks are drained —
            // the current request holds nothing yet, and in-flight prefetches
            // keep their reserved blocks, so this never touches live KV.
            self.kv_mgr.pool().evict_inactive();
        }
        let probe = self
            .kv_mgr
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_adapter);
        let query_hashes = probe.cpu_query_hashes();
        if query_hashes.is_empty() {
            return false;
        }
        let hit = match offload.query(&request_id.0.to_string(), &query_hashes) {
            Ok(hit) => hit,
            Err(e) => {
                log::warn!("KV offload query failed for {request_id:?} (skipping): {e}");
                return false;
            }
        };
        let (Some(lease), num_blocks) = (hit.lease, hit.num_blocks) else {
            return false; // miss
        };
        // Blocks promised to admitted requests are off-limits: reserving into
        // them makes a later prefill chunk or decode growth fail allocation.
        if self
            .kv_mgr
            .pool()
            .available_blocks()
            .saturating_sub(reserve_floor)
            < num_blocks
        {
            offload.release_query_lease(lease);
            return false;
        }
        let Some(reservation) = self.kv_mgr.pool().reserve_loaded_blocks(num_blocks) else {
            // Block pressure: release the lease so its pinned host blocks aren't
            // held for the full lease TTL, and prefill from scratch rather than
            // stall.
            offload.release_query_lease(lease);
            return false;
        };
        let page_ids = reservation.page_ids();
        let handle = match offload.load(lease, page_ids) {
            Ok(handle) => handle,
            Err(e) => {
                log::warn!("KV offload load submit failed for {request_id:?} (skipping): {e}");
                // `load` consumes the lease only past its early validation; a
                // submit error may leave it pinned, so release it (no-op if it
                // was already consumed).
                offload.release_query_lease(lease);
                return false;
            }
        };
        self.prefetch.insert(
            request_id,
            super::PrefetchState {
                probe,
                reservation: Some(reservation),
                handle: Some(handle),
            },
        );
        true
    }

    fn drain_ready_prefetch(&mut self) -> Vec<RequestId> {
        let ids: Vec<RequestId> = self.prefetch.keys().copied().collect();
        let mut done = Vec::new();
        for id in ids {
            let poll = match self.prefetch.get_mut(&id).and_then(|st| st.handle.as_mut()) {
                Some(handle) => handle.poll(),
                None => continue, // already settled, awaiting prefill
            };
            if let Some(result) = poll {
                self.settle_prefetch(id, result);
                done.push(id);
            }
        }
        done
    }

    fn wait_ready_prefetch(&mut self) -> Vec<RequestId> {
        let mut done = Vec::new();
        if let Some(id) = self
            .prefetch
            .iter()
            .find(|(_, st)| st.handle.is_some())
            .map(|(id, _)| *id)
        {
            let handle = self
                .prefetch
                .get_mut(&id)
                .and_then(|st| st.handle.take())
                .expect("in-flight handle present");
            let result = handle.wait();
            self.settle_prefetch(id, result);
            // `settle_prefetch` clears the handle, so the drain below skips it;
            // record it here as the one we blocked on.
            done.push(id);
        }
        // Sweep any others that completed concurrently.
        for id in self.drain_ready_prefetch() {
            if !done.contains(&id) {
                done.push(id);
            }
        }
        done
    }

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        // 1. Create RequestKvs (first chunk only), clamp chunk budgets,
        // schedule KV for this step's tokens
        let mut requests = plan.requests.to_vec();
        for req in &mut requests {
            self.schedule_prefill_chunk(req)?;
        }

        // 2. Build KvViews (seq_len = chunk_start + this chunk)
        let kv_views: Vec<KvView> = requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.chunk_tokens))
            .collect();

        // 3. Execute forward
        let scheduled_requests = requests.clone();
        let step = StepCommand::Prefill {
            requests,
            kv_views,
            echo: plan.echo,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply prefill
        let result = match outcome {
            WorkerStepOutcome::Prefill(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "prefill returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            self.apply_prefill_result(req_result)?;
        }
        if self.speculative_enabled {
            for (req, req_result) in scheduled_requests.iter().zip(&result.requests) {
                assert_eq!(req.request_id, req_result.request_id);
                match dflash_prefill_action(result.dflash_context_captured, req_result.completed) {
                    DFlashPrefillAction::MarkReady => {
                        self.dflash_ready_requests.insert(req.request_id);
                    }
                    DFlashPrefillAction::KeepPending => {
                        self.dflash_ready_requests.remove(&req.request_id);
                    }
                    DFlashPrefillAction::Drop => {
                        self.dflash_ready_requests.remove(&req.request_id);
                        self.primary.drop_dflash_request(req.request_id)?;
                    }
                }
            }
        }
        // 5. Offload the blocks this prefill just sealed (post-step-sync).
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        // 1. Schedule decode for all active requests
        for req in plan.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(self.kv_mgr.pool()).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let kv_views: Vec<KvView> = plan
            .requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward
        let scheduled_requests = plan.requests.to_vec();
        let step = StepCommand::Decode {
            requests: scheduled_requests.clone(),
            kv_views,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply decode
        let result = match outcome {
            WorkerStepOutcome::Decode(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "decode returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after decode");
            rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
        }
        if self.speculative_enabled {
            for req in &scheduled_requests {
                self.dflash_ready_requests.remove(&req.request_id);
                self.primary.drop_dflash_request(req.request_id)?;
            }
        }
        // 5. Offload any block this decode step just sealed (post-step-sync).
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn execute_speculative_verify(
        &mut self,
        plan: SpeculativeVerifyPlan<'_>,
    ) -> Result<SpeculativeVerifyResult> {
        self.execute_speculative_verify_impl(plan)
    }

    fn execute_speculative_draft(
        &mut self,
        plan: SpeculativeDraftPlan<'_>,
    ) -> Result<SpeculativeDraftResult> {
        self.execute_speculative_draft_impl(plan)
    }

    fn speculative_enabled(&self) -> bool {
        self.speculative_enabled
    }

    fn speculative_request_ready(&self, request_id: RequestId) -> bool {
        self.dflash_ready_requests.contains(&request_id)
    }

    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        // 1. Create RequestKvs for prefill requests (first chunk only), clamp
        // chunk budgets, schedule KV for this step's tokens
        let mut prefill_requests = plan.prefill_requests.to_vec();
        for req in &mut prefill_requests {
            self.schedule_prefill_chunk(req)?;
        }

        // Schedule decode for active requests
        let decode_requests = plan.decode_requests.to_vec();
        for req in &decode_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(self.kv_mgr.pool()).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let prefill_kv_views: Vec<KvView> = prefill_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.chunk_tokens))
            .collect();
        let decode_kv_views: Vec<KvView> = decode_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward
        let step = StepCommand::Unified {
            prefill_requests,
            prefill_kv_views,
            decode_requests,
            decode_kv_views,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply both prefill and decode
        let result = match outcome {
            WorkerStepOutcome::Unified(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "unified returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.prefill_requests {
            self.apply_prefill_result(req_result)?;
        }
        for req_result in &result.decode_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after unified decode");
            rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
        }
        if self.speculative_enabled {
            for req_result in &result.prefill_requests {
                self.dflash_ready_requests.remove(&req_result.request_id);
                self.primary.drop_dflash_request(req_result.request_id)?;
            }
            for req_result in &result.decode_requests {
                self.dflash_ready_requests.remove(&req_result.request_id);
                self.primary.drop_dflash_request(req_result.request_id)?;
            }
        }
        // 5. Offload sealed blocks from both halves (post-step-sync).
        for req_result in &result.prefill_requests {
            self.save_sealed_blocks(req_result.request_id);
        }
        for req_result in &result.decode_requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        ensure_lora_capacity(
            &self.loaded_lora_adapters,
            &request.lora_name,
            self.lora_options.max_loras,
            request.load_inplace,
        )?;
        let adapter = crate::lora::load_lora_adapter(
            &request.lora_path,
            &self.metadata.config,
            self.lora_options.max_lora_rank,
        )?;
        let world_size = self.workers.len() + 1;
        let projection_count: usize = adapter
            .layers
            .iter()
            .map(|layer| layer.projections.len())
            .sum();
        let element_count: usize = adapter
            .layers
            .iter()
            .flat_map(|layer| layer.projections.values())
            .map(|projection| projection.a.data.len() + projection.b.data.len())
            .sum();
        let shape_elems: usize = adapter
            .layers
            .iter()
            .flat_map(|layer| layer.projections.values())
            .map(|projection| {
                projection.a.rows * projection.a.cols + projection.b.rows * projection.b.cols
            })
            .sum();
        debug_assert_eq!(element_count, shape_elems);
        let rank = adapter.manifest.rank;
        let targets = adapter.manifest.target_modules.join(", ");
        let path = adapter.manifest.path.display().to_string();
        let mut sharded_adapters = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            sharded_adapters.push(adapter.shard_for_tensor_parallel(
                &self.metadata.config,
                TensorParallelConfig { rank, world_size },
            )?);
        }

        let mut sharded_adapters = sharded_adapters.into_iter();
        let primary_adapter = sharded_adapters
            .next()
            .expect("rank 0 adapter must exist for nonzero world_size");
        let primary_response = self.primary.load_lora_adapter(
            request.lora_name.clone(),
            primary_adapter,
            request.load_inplace,
        )?;
        let mut pending = Vec::with_capacity(self.workers.len());
        let mut errors = Vec::new();
        for (index, worker) in self.workers.iter().enumerate() {
            let rank = index + 1;
            let rank_adapter = sharded_adapters
                .next()
                .expect("worker adapter must exist for every tensor-parallel rank");
            match worker.load_lora_adapter(
                request.lora_name.clone(),
                rank_adapter,
                request.load_inplace,
            ) {
                Ok(response) => pending.push((rank, response)),
                Err(err) => errors.push(format!("rank {rank} dispatch: {err:#}")),
            }
        }

        match primary_response.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("rank 0: {err:#}")),
            Err(_) => errors.push("rank 0: dropped LoRA load response".to_string()),
        }
        for (rank, response) in pending {
            match response.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("rank {rank}: {err:#}")),
                Err(_) => errors.push(format!("rank {rank}: dropped LoRA load response")),
            }
        }
        if !errors.is_empty() {
            let mut cleanup_errors = Vec::new();
            match self.primary.discard_lora_adapter(request.lora_name.clone()) {
                Ok(response) => match response.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => cleanup_errors.push(format!("rank 0 cleanup: {err:#}")),
                    Err(_) => cleanup_errors
                        .push("rank 0 cleanup: dropped LoRA discard response".to_string()),
                },
                Err(err) => cleanup_errors.push(format!("rank 0 cleanup dispatch: {err:#}")),
            }
            for (index, worker) in self.workers.iter().enumerate() {
                let rank = index + 1;
                match worker.discard_lora_adapter(request.lora_name.clone()) {
                    Ok(response) => match response.recv() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            cleanup_errors.push(format!("rank {rank} cleanup: {err:#}"));
                        }
                        Err(_) => cleanup_errors.push(format!(
                            "rank {rank} cleanup: dropped LoRA discard response"
                        )),
                    },
                    Err(err) => {
                        cleanup_errors.push(format!("rank {rank} cleanup dispatch: {err:#}"));
                    }
                }
            }
            if cleanup_errors.is_empty() {
                self.loaded_lora_adapters.remove(&request.lora_name);
            }
            let cleanup_suffix = if cleanup_errors.is_empty() {
                String::new()
            } else {
                format!("; cleanup errors: {}", cleanup_errors.join("; "))
            };
            anyhow::bail!(
                "failed to load Qwen3 LoRA adapter {} on tensor-parallel ranks: {}{}",
                request.lora_name,
                errors.join("; "),
                cleanup_suffix
            );
        }

        log::info!(
            "Loaded Qwen3 LoRA adapter {} from {} (rank={}, targets={}, projections={}, bf16_elements={}, tp_world_size={}, load_inplace={})",
            request.lora_name,
            path,
            rank,
            targets,
            projection_count,
            element_count,
            world_size,
            request.load_inplace
        );
        self.loaded_lora_adapters.insert(request.lora_name.clone());
        Ok(())
    }

    fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
        let primary_response = self
            .primary
            .unload_lora_adapter(request.lora_name.clone())?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for (index, worker) in self.workers.iter().enumerate() {
            pending.push((
                index + 1,
                worker.unload_lora_adapter(request.lora_name.clone())?,
            ));
        }

        let mut errors = Vec::new();
        match primary_response.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("rank 0: {err:#}")),
            Err(_) => errors.push("rank 0: dropped LoRA unload response".to_string()),
        }
        for (rank, response) in pending {
            match response.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("rank {rank}: {err:#}")),
                Err(_) => errors.push(format!("rank {rank}: dropped LoRA unload response")),
            }
        }
        if !errors.is_empty() {
            anyhow::bail!(
                "failed to unload Qwen3 LoRA adapter {} on tensor-parallel ranks: {}",
                request.lora_name,
                errors.join("; ")
            );
        }

        log::info!("Unloaded Qwen3 LoRA adapter {}", request.lora_name);
        self.loaded_lora_adapters.remove(&request.lora_name);
        Ok(())
    }

    fn list_lora_adapters(&self) -> Vec<String> {
        let mut names: Vec<_> = self.loaded_lora_adapters.iter().cloned().collect();
        names.sort();
        names
    }
}

fn ensure_lora_capacity(
    loaded_lora_adapters: &HashSet<String>,
    lora_name: &str,
    max_loras: usize,
    load_inplace: bool,
) -> Result<()> {
    if loaded_lora_adapters.contains(lora_name) {
        anyhow::ensure!(
            load_inplace,
            "Qwen3 LoRA adapter {lora_name} is already loaded"
        );
        return Ok(());
    }
    anyhow::ensure!(
        loaded_lora_adapters.len() < max_loras,
        "Qwen3 LoRA adapter capacity exceeded: max_loras={}, loaded_adapters={}, requested={}",
        max_loras,
        loaded_lora_adapters.len(),
        lora_name
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ensure_lora_capacity;
    use std::collections::HashSet;

    #[test]
    fn lora_capacity_rejects_new_adapter_at_limit() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        let error = ensure_lora_capacity(&loaded, "adapter-b", 1, false)
            .expect_err("new adapter should exceed capacity")
            .to_string();

        assert!(error.contains("max_loras=1"));
        assert!(error.contains("requested=adapter-b"));
    }

    #[test]
    fn lora_capacity_allows_existing_adapter_replacement_at_limit_with_load_inplace() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        ensure_lora_capacity(&loaded, "adapter-a", 1, true)
            .expect("existing adapter should fit with load_inplace");
    }

    #[test]
    fn lora_capacity_rejects_duplicate_without_load_inplace() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        let error = ensure_lora_capacity(&loaded, "adapter-a", 1, false)
            .expect_err("duplicate without load_inplace should fail")
            .to_string();

        assert!(error.contains("already loaded"));
    }
}
