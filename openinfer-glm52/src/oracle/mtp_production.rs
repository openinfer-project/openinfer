//! Production-path regression for the target-hidden boundary consumed by MTP.

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::TokenEvent;
use openinfer_core::engine::TokenSink;
use openinfer_sample::SamplingParams;

use crate::Glm52LaunchOptions;
use crate::Glm52MoeTopo;

const PATHOLOGICAL_PROMPT: &[u32] = &[
    98770, 98771, 98772, 98773, 98774, 98775, 98776, 98777, 98778, 98779, 98780, 98781, 98782,
    98783, 98784, 98785, 98786, 98787, 98788, 98789, 98790, 98791, 98792, 98793, 98794, 98795,
    98796, 98797, 98798, 98799, 98800, 98801, 98802, 98803, 98804, 98805, 98806, 98807, 98808,
    98809, 98810, 98811, 98812, 98813, 98814, 98815, 98816, 98817, 98818, 98819, 98820, 98821,
    98822, 98823, 98824, 98825, 98826, 98827, 5691, 109691, 98831, 98832, 98833, 98834, 98835,
    98836, 98837, 98838, 98839, 98840, 98841, 98842, 5691, 98844, 98845, 98846, 98847, 5691, 98849,
    98850, 98851, 98852, 98853, 98854, 5691, 98856, 98857, 98858, 98859, 98860, 98861, 98862,
    98863, 98864, 98865, 98866, 98867, 98868, 98869, 98870, 98871, 98872, 98873, 98874, 98875,
    98876, 98877, 98878, 98879, 98880, 98881, 98882, 98883, 98884, 98885, 98886, 98887, 98888,
    98889, 98890, 98891, 98892, 98893, 98894, 98895, 98896, 98897,
];

#[test]
#[ignore = "requires 8×H200 + GLM-5.2-FP8 checkpoint + NCCL >= 2.30.4"]
fn native_mtp_uses_final_normalized_target_hidden() -> Result<()> {
    let model_path = std::env::var_os("OPENINFER_TEST_MODEL_PATH")
        .map(PathBuf::from)
        .context("OPENINFER_TEST_MODEL_PATH must point to GLM-5.2-FP8")?;
    crate::scheduler::reset_mtp_production_stats();
    let engine = crate::launch(
        &model_path,
        Glm52LaunchOptions {
            tp_size: 1,
            dp_size: 8,
            dspark_draft_model_path: None,
            native_mtp: true,
            max_model_len: Some(4096),
            prefill_only: None,
            no_prefix_cache: true,
            kv_offload: None,
            moe_topo: Glm52MoeTopo::Ep8,
            weight_staging: true,
            dump_graph_png: None,
            rank_hosts: Vec::new(),
        },
    )?;
    let (token_tx, mut token_rx) = TokenSink::standalone();
    engine.submit(GenerateRequest {
        request_id: Some(crate::scheduler::MTP_PRODUCTION_GATE_REQUEST_ID.into()),
        queued_at_unix_s: None,
        trace_parent: None,
        data_parallel_rank: Some(0),
        prompt_tokens: PATHOLOGICAL_PROMPT.to_vec(),
        params: SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens: 256,
        lora_adapter: None,
        token_tx,
        logprobs: 0,
        echo: false,
    })?;

    let mut completion = Vec::new();
    loop {
        let (_, event) = token_rx
            .blocking_recv()
            .context("GLM5.2 engine closed the production-gate token stream")?;
        match event {
            TokenEvent::Token { id, .. } => completion.push(id),
            TokenEvent::Finished {
                completion_tokens, ..
            } => {
                assert_eq!(completion_tokens, 256);
                break;
            }
            TokenEvent::Error { message, .. } | TokenEvent::Rejected { message, .. } => {
                anyhow::bail!("GLM5.2 production gate failed: {message}")
            }
            TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. } => {}
        }
    }
    drop(engine);

    assert_eq!(completion.len(), 256);
    assert!(
        completion.iter().all(|&token| token == 98824),
        "selected target trajectory changed: {:?}",
        &completion[..completion.len().min(16)]
    );
    let stats = crate::scheduler::mtp_production_stats();
    let first_proposal = stats
        .first_proposal
        .as_deref()
        .context("native MTP did not produce a proposal")?;
    assert_eq!(first_proposal.first(), Some(&98825));
    assert!(stats.rounds > 0, "native MTP did not verify any proposal");
    let mean_accepted_length = 1.0 + stats.accepted_drafts as f64 / stats.rounds as f64;
    assert!(
        mean_accepted_length >= 5.0,
        "native MTP mean accepted length regressed to {mean_accepted_length:.3}: {stats:?}"
    );
    Ok(())
}
