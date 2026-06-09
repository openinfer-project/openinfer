//! Qwen3.5 batch prefill and unified step (prefill + decode combined).
//!
//! Linear attention (GDR chunkwise) does not have an efficient batched prefill
//! kernel, so `batch_prefill` runs each request's prefill serially. Full-attention
//! layers also run per-request to reuse the existing paged prefill path.
//!
//! `unified_step` combines:
//!   1. Serial `batch_prefill` for new requests entering the batch.
//!   2. CUDA Graph `batch_decode_graph` for existing decode requests.

use anyhow::Result;

use super::batch_decode_graph::BatchDecodeGraphState;
use super::recurrent_state::RecurrentState;
use super::weights::Qwen35Model;
use crate::ops;
use openinfer_core::kv_pool::KvState;
use openinfer_core::tensor::DeviceVec;

impl Qwen35Model {
    /// Prefill `n` prompts sequentially, updating each request's KV and recurrent state.
    ///
    /// Returns one `DeviceVec` of logits (vocab_size) per request (last-token logits).
    /// Requests are independent — there is no cross-request batching in the prefill pass.
    pub(crate) fn batch_prefill(
        &self,
        prompts: &[&[u32]],
        kv_states: &mut [KvState],
        recurrent_states: &mut [&mut RecurrentState],
    ) -> Result<Vec<DeviceVec>> {
        let n = prompts.len();
        anyhow::ensure!(n > 0, "batch_prefill requires at least one prompt");
        anyhow::ensure!(n == kv_states.len(), "prompts / kv_states len mismatch");
        anyhow::ensure!(
            n == recurrent_states.len(),
            "prompts / recurrent_states len mismatch"
        );

        let mut logits = Vec::with_capacity(n);
        for i in 0..n {
            let logit = self.prefill_forward(prompts[i], &mut kv_states[i], recurrent_states[i])?;
            logits.push(logit);
        }
        Ok(logits)
    }

    /// Unified step: prefill new requests and decode existing requests in one call.
    ///
    /// Prefill is run serially per-request (GDR chunkwise per request). Decode runs
    /// via CUDA Graph on the pre-allocated `graph_state` for the decode batch.
    ///
    /// Either `prefill_prompts` or `decode_tokens` may be empty (but not both).
    ///
    /// Returns `(prefill_logits, decode_logits)` — one `DeviceVec` per request.
    /// Decode logits are extracted from `graph_state.buffers.logits` after the decode
    /// forward pass; callers can use them for sampling.
    pub(crate) fn unified_step(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_kv_states: &mut [KvState],
        prefill_recurrent_states: &mut [&mut RecurrentState],
        decode_tokens: &[u32],
        decode_kv_states: &mut [&mut KvState],
        graph_state: &mut BatchDecodeGraphState,
    ) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
        anyhow::ensure!(
            !prefill_prompts.is_empty() || !decode_tokens.is_empty(),
            "unified_step: both prefill and decode are empty"
        );

        // ── Prefill phase ─────────────────────────────────────────────────────
        let prefill_logits = if prefill_prompts.is_empty() {
            Vec::new()
        } else {
            self.batch_prefill(prefill_prompts, prefill_kv_states, prefill_recurrent_states)?
        };

        // ── Decode phase ──────────────────────────────────────────────────────
        let decode_logits = if decode_tokens.is_empty() {
            Vec::new()
        } else {
            self.batch_decode_graph(decode_tokens, decode_kv_states, graph_state)?;

            // Extract per-request logits from the batch logits buffer.
            let num_decode = decode_tokens.len();
            let mut dlogits = Vec::with_capacity(num_decode);
            for i in 0..num_decode {
                let logit = ops::extract_vec(&self.ctx, &graph_state.buffers.logits, i)?;
                dlogits.push(logit);
            }
            dlogits
        };

        Ok((prefill_logits, decode_logits))
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::path::Path;

    use super::*;
    use openinfer_core::kv_pool::KvState;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");

    fn get_model_path_or_skip() -> Option<String> {
        match std::env::var("OPENINFER_TEST_MODEL_PATH") {
            Ok(path) => Some(path),
            Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
                Some(MODEL_PATH.to_string())
            }
            Err(_) => {
                eprintln!(
                    "skipping Qwen3.5 unified forward model test because {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
                );
                None
            }
        }
    }

    /// Sample a token from a DeviceVec logits using greedy (argmax).
    fn greedy_sample(model: &Qwen35Model, logits: &DeviceVec, _rng: &mut StdRng) -> u32 {
        let params = openinfer_core::sampler::SamplingParams::default();
        let mut probs: cudarc::driver::CudaSlice<f32> = model
            .ctx
            .stream
            .alloc_zeros(model.config.vocab_size)
            .unwrap();
        let mut top1_value: cudarc::driver::CudaSlice<half::bf16> =
            model.ctx.stream.alloc_zeros(1).unwrap();
        let mut row_states: cudarc::driver::CudaSlice<u8> = model
            .ctx
            .stream
            .alloc_zeros(crate::ops::flashinfer_topk_row_states_bytes())
            .unwrap();
        let mut valid: cudarc::driver::CudaSlice<u8> = model.ctx.stream.alloc_zeros(1).unwrap();
        let mut out: cudarc::driver::CudaSlice<i32> = model.ctx.stream.alloc_zeros(1).unwrap();
        crate::ops::gpu_sample_into(
            &model.ctx,
            logits,
            &mut probs,
            &mut top1_value,
            &mut row_states,
            &mut valid,
            &mut out,
            &params,
            0.0,
        )
        .unwrap()
    }

    /// Verify that unified_step decode output matches batch_decode_graph standalone.
    #[test]
    fn unified_step_decode_matches_graph_decode() {
        let Some(model_path) = get_model_path_or_skip() else {
            return;
        };
        let model = Qwen35Model::from_safetensors_with_options(&model_path, true).unwrap();

        let prompt_a: Vec<u32> = vec![9707];
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
        let num_steps = 5;
        let seed = 42;

        // --- Reference: standalone batch_decode_graph ---
        let ref_tokens = {
            let mut kv_a = model.alloc_kv();
            let mut kv_b = model.alloc_kv();
            let mut rec_a = RecurrentState::new(&model.ctx, &model.config).unwrap();
            let mut rec_b = RecurrentState::new(&model.ctx, &model.config).unwrap();

            // Prefill both prompts with separate request-local sequence trackers.
            let first_logits_a = model
                .prefill_forward(&prompt_a, &mut kv_a, &mut rec_a)
                .unwrap();
            let first_logits_b = model
                .prefill_forward(&prompt_b, &mut kv_b, &mut rec_b)
                .unwrap();

            let mut rng = StdRng::seed_from_u64(seed);
            let first_a = greedy_sample(&model, &first_logits_a, &mut rng);
            let first_b = greedy_sample(&model, &first_logits_b, &mut rng);

            let mut gs = model
                .create_batch_decode_graph_state_with_capacity(2)
                .unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_a, 0).unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_b, 1).unwrap();
            model.ctx.sync().unwrap();

            let mut tokens_a = vec![first_a];
            let mut tokens_b = vec![first_b];
            let mut kv_refs: Vec<&mut KvState> = vec![&mut kv_a, &mut kv_b];

            for _ in 1..num_steps {
                let tids = [*tokens_a.last().unwrap(), *tokens_b.last().unwrap()];
                model
                    .batch_decode_graph(&tids, &mut kv_refs, &mut gs)
                    .unwrap();
                let la = ops::extract_vec(&model.ctx, &gs.buffers.logits, 0).unwrap();
                let lb = ops::extract_vec(&model.ctx, &gs.buffers.logits, 1).unwrap();
                tokens_a.push(greedy_sample(&model, &la, &mut rng));
                tokens_b.push(greedy_sample(&model, &lb, &mut rng));
            }
            (tokens_a, tokens_b)
        };

        // --- unified_step path ---
        let unified_tokens = {
            let prompts_ref: Vec<&[u32]> = vec![&prompt_a, &prompt_b];
            let mut kv_prefill: Vec<KvState> = vec![model.alloc_kv(), model.alloc_kv()];
            let mut rec_p: Vec<RecurrentState> = vec![
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
            ];
            let mut rec_p_refs: Vec<&mut RecurrentState> = rec_p.iter_mut().collect();

            let (prefill_logits, _) = model
                .unified_step(
                    &prompts_ref,
                    &mut kv_prefill,
                    &mut rec_p_refs,
                    &[],
                    &mut [],
                    &mut model
                        .create_batch_decode_graph_state_with_capacity(2)
                        .unwrap(),
                )
                .unwrap();

            let mut rng = StdRng::seed_from_u64(seed);
            let first_a = greedy_sample(&model, &prefill_logits[0], &mut rng);
            let first_b = greedy_sample(&model, &prefill_logits[1], &mut rng);

            // Transfer prefill states to decode graph slots
            let mut gs = model
                .create_batch_decode_graph_state_with_capacity(2)
                .unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_p[0], 0).unwrap();
            gs.copy_state_to_slot(&model.ctx, &rec_p[1], 1).unwrap();

            let mut kv_refs: Vec<&mut KvState> = kv_prefill.iter_mut().collect();

            let mut tokens_a = vec![first_a];
            let mut tokens_b = vec![first_b];

            for _ in 1..num_steps {
                let tids = [*tokens_a.last().unwrap(), *tokens_b.last().unwrap()];
                let (_, dlogits) = model
                    .unified_step(&[], &mut [], &mut [], &tids, &mut kv_refs, &mut gs)
                    .unwrap();
                tokens_a.push(greedy_sample(&model, &dlogits[0], &mut rng));
                tokens_b.push(greedy_sample(&model, &dlogits[1], &mut rng));
            }
            (tokens_a, tokens_b)
        };

        assert_eq!(
            unified_tokens, ref_tokens,
            "unified_step decode mismatch:\n  unified: {:?}\n  ref:     {:?}",
            unified_tokens, ref_tokens
        );
    }
}
