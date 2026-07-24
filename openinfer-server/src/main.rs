mod config;

use std::time::Instant;

use anyhow::Context;
use clap::CommandFactory;
use clap::FromArgMatches;
use config::Args;
use log::info;
use openinfer::logging;
use openinfer::server_engine::ModelType;
use openinfer::server_engine::detect_model_type;
use openinfer_core::engine::EngineHandle;
#[cfg(feature = "qwen3")]
use openinfer_qwen3::Qwen3LaunchOptions;
#[cfg(feature = "qwen3")]
use openinfer_qwen3::Qwen3LoraOptions;
#[cfg(feature = "qwen3")]
use openinfer_qwen3::Qwen3OffloadOptions;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init_default();
    openinfer_core::tracing::init();

    let matches = Args::command().get_matches();
    let args =
        Args::from_arg_matches(&matches).map_err(|e| anyhow::anyhow!("invalid CLI args: {e}"))?;
    let provided = config::provided_args(&matches);

    // rank-host mode: a dumb worker shell for a remote GLM5.2 coordinator —
    // no engine, no HTTP. Serves connections until killed.
    if let Some(listen) = &args.glm52_rank_host {
        #[cfg(feature = "glm52")]
        {
            return tokio::task::spawn_blocking({
                let listen = listen.clone();
                move || openinfer_glm52::serve_rank_host(&listen)
            })
            .await
            .context("rank-host thread panicked")?;
        }
        #[cfg(not(feature = "glm52"))]
        anyhow::bail!("--glm52-rank-host requires the glm52 feature (got {listen})");
    }

    let model_type = detect_model_type(&args.model_path).with_context(|| {
        format!(
            "failed to detect model type from {}",
            args.model_path.display()
        )
    })?;
    args.validate(model_type, &provided)?;

    info!("=== openinfer - {} (GPU) ===", model_type);
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime: model_path={}, user-set flags {provided:?}",
        args.model_path.display(),
    );

    // Engine load (weights → GPU) runs on a blocking thread so the HTTP
    // frontend (tokenizer, chat templates) loads concurrently. The frontend
    // binds only after the engine registers, so readiness is unchanged.
    let model_path = args.model_path.clone();
    let served_model_name = args.served_model_name.clone();
    let lora_modules = args.lora_modules.clone();
    let enable_lora = args.enable_lora;
    let port = args.port;
    let frontend_engine_count = 1;
    #[cfg(feature = "glm52")]
    let glm52_prefill_only = model_type == ModelType::Glm52 && args.glm52_prefill_only;
    #[cfg(not(feature = "glm52"))]
    let glm52_prefill_only = false;
    #[cfg(feature = "glm52")]
    let frontend_engine_count = if model_type == ModelType::Glm52 {
        args.moe_topo
            .parse::<openinfer_glm52::Glm52MoeTopo>()
            .context("--moe-topo")?
            .logical_rank_count()
    } else {
        frontend_engine_count
    };
    let engine_load = tokio::task::spawn_blocking(move || -> anyhow::Result<EngineHandle> {
        load_engine(&args, model_type)
    });

    let serve_result = if enable_lora {
        // LoRA routes need the engine handle when the router is built, so this
        // path stays sequential.
        let handle = engine_load
            .await
            .context("engine loader thread panicked")??;
        info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());
        let max_model_len =
            openinfer::vllm_frontend::load_max_model_len(&model_path).unwrap_or(4096);
        openinfer::vllm_frontend::serve_model_with_lora_routes(
            handle,
            model_path.to_string_lossy().into_owned(),
            served_model_name.into_iter().collect(),
            lora_modules,
            port,
            max_model_len,
            openinfer::vllm_frontend::shutdown_token_from_ctrl_c(),
        )
        .await
    } else {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let engine = {
            let shutdown = shutdown.clone();
            async move {
                let handle = engine_load
                    .await
                    .context("engine loader thread panicked")??;
                info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());
                // The blocking load can't be cancelled, so SIGINT keeps its
                // default kill behavior until the engine is up; only then
                // switch to graceful shutdown.
                openinfer::vllm_frontend::cancel_token_on_ctrl_c(&shutdown);
                anyhow::Ok(handle)
            }
        };
        if glm52_prefill_only {
            openinfer::vllm_frontend::serve_prefill_only_with_engine_count(
                engine,
                &model_path,
                served_model_name.into_iter().collect(),
                port,
                None,
                frontend_engine_count,
                shutdown,
            )
            .await
        } else {
            openinfer::vllm_frontend::serve_with_engine_count(
                engine,
                &model_path,
                served_model_name.into_iter().collect(),
                port,
                None,
                frontend_engine_count,
                shutdown,
            )
            .await
        }
    }
    .context("vLLM frontend server failed");

    // Export the final batch of request spans before the runtime tears down.
    // Flush before propagating an error too — a failed server is exactly where
    // the last buffered spans matter. No-op when tracing was never enabled.
    openinfer_core::tracing::flush();
    serve_result?;

    Ok(())
}

// Pure dispatch: each model crate owns its own launch policy (topology
// defaults, capability constraints, cross-arg validation). The server only
// picks the crate by detected model type and forwards the relevant CLI knobs.
fn load_engine(args: &Args, model_type: ModelType) -> anyhow::Result<EngineHandle> {
    let handle = match model_type {
        #[cfg(feature = "deepseek-v2-lite")]
        ModelType::DeepSeekV2Lite => {
            openinfer_deepseek_v2_lite::launch(&args.model_path, args.cuda_graph)
                .context("failed to start DeepSeek V2 Lite engine")?
        }
        #[cfg(feature = "glm52")]
        ModelType::Glm52 => {
            let moe_topo: openinfer_glm52::Glm52MoeTopo =
                args.moe_topo.parse().context("--moe-topo")?;
            openinfer_glm52::launch(
                &args.model_path,
                openinfer_glm52::Glm52LaunchOptions {
                    tp_size: args.tp_size,
                    dp_size: args.dp_size.unwrap_or_else(|| moe_topo.default_dp_size()),
                    dspark_draft_model_path: args.dflash_draft_model_path.clone(),
                    max_model_len: args.max_model_len,
                    prefill_only: args.glm52_prefill_only.then_some(
                        openinfer_glm52::Glm52PrefillOnlyOptions {
                            chunk_size: args.glm52_prefill_chunk_size,
                        },
                    ),
                    no_prefix_cache: args.no_prefix_cache,
                    kv_offload: args
                        .kv_offload
                        .then(|| openinfer_glm52::Glm52KvOffloadOptions {
                            pinned_pool_bytes: (args.kv_offload_host_gib * f64::from(1u32 << 30))
                                as usize,
                            use_hugepages: args.kv_offload_hugepages,
                            p2p: match (
                                args.kv_p2p_metaserver_addr.clone(),
                                args.kv_p2p_advertise_addr.clone(),
                            ) {
                                (Some(metaserver_addr), Some(advertise_addr)) => {
                                    Some(openinfer_glm52::Glm52P2pOptions {
                                        metaserver_addr,
                                        advertise_addr,
                                        rdma_nics: args.kv_p2p_nics.clone(),
                                    })
                                }
                                _ => None,
                            },
                            vllm_compat: args.kv_pd_vllm_seed.clone().map(|seed| {
                                openinfer_glm52::Glm52VllmCompatOptions {
                                    python_hash_seed: seed,
                                    namespace: args
                                        .kv_pd_vllm_namespace
                                        .clone()
                                        .expect("clap requires kv_pd_vllm_namespace"),
                                    miss_wait: std::time::Duration::from_millis(
                                        args.kv_pd_miss_wait_ms,
                                    ),
                                    allow_local_prefill: args.kv_pd_allow_local_prefill,
                                }
                            }),
                        }),
                    moe_topo,
                    weight_staging: args.glm52_weight_staging,
                    dump_graph_png: args.dump_graph_png.clone(),
                    rank_hosts: args
                        .rank_hosts
                        .iter()
                        .map(|spec| spec.parse())
                        .collect::<anyhow::Result<Vec<_>>>()
                        .context("--rank-hosts")?,
                },
            )
            .context("failed to start GLM5.2 engine")?
        }
        #[cfg(feature = "kimi-k2")]
        ModelType::KimiK2 => openinfer_kimi_k2::launch(
            &args.model_path,
            openinfer_kimi_k2::KimiLaunchOptions {
                tp_size: args.tp_size,
                dp_size: args.dp_size.unwrap_or(8),
                ep_backend: args.ep_backend.into(),
                cuda_graph: args.cuda_graph,
            },
        )
        .context("failed to start Kimi-K2.6 text engine")?,
        #[cfg(feature = "qwen3")]
        ModelType::Qwen3 => {
            let offload = if args.kv_offload {
                let bytes = (args.kv_offload_host_gib * f64::from(1u32 << 30)) as usize;
                let mut offload = Qwen3OffloadOptions::enabled(bytes);
                offload.use_hugepages = args.kv_offload_hugepages;
                if let (Some(metaserver_addr), Some(advertise_addr)) = (
                    args.kv_p2p_metaserver_addr.clone(),
                    args.kv_p2p_advertise_addr.clone(),
                ) {
                    offload = offload.with_p2p(openinfer_qwen3::Qwen3P2pOptions {
                        metaserver_addr,
                        advertise_addr,
                        rdma_nics: args.kv_p2p_nics.clone(),
                        flush_on_finish: args.kv_p2p_flush_on_finish,
                    });
                }
                if let Some(seed) = args.kv_pd_vllm_seed.clone() {
                    offload = offload.with_vllm_compat(openinfer_qwen3::Qwen3VllmCompatOptions {
                        python_hash_seed: seed,
                        namespace: args
                            .kv_pd_vllm_namespace
                            .clone()
                            .expect("clap requires kv_pd_vllm_namespace"),
                        miss_wait: std::time::Duration::from_millis(args.kv_pd_miss_wait_ms),
                    });
                }
                offload
            } else {
                Qwen3OffloadOptions::disabled()
            };
            let lora = args.enable_lora.then_some(Qwen3LoraOptions {
                max_loras: args.max_loras,
                max_lora_rank: args.max_lora_rank,
            });
            let kv_cache_memory_margin_bytes = args
                .kv_cache_memory_margin_mib
                .checked_mul(1 << 20)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "--kv-cache-memory-margin-mib is too large: {}",
                        args.kv_cache_memory_margin_mib
                    )
                })?;
            let dflash_draft_model_path = match args.dflash_draft_model_path.clone() {
                Some(path) => {
                    anyhow::ensure!(
                        !args.enable_lora,
                        "--dflash-draft-model-path is not supported with --enable-lora"
                    );
                    anyhow::ensure!(
                        !args.kv_offload,
                        "--dflash-draft-model-path is not supported with --kv-offload"
                    );
                    anyhow::ensure!(
                        args.tp_size == 1,
                        "--dflash-draft-model-path currently requires --tp-size=1"
                    );
                    Some(path)
                }
                None => None,
            };
            openinfer_qwen3::launch(
                &args.model_path,
                Qwen3LaunchOptions {
                    device_ordinal: args.device_ordinal,
                    tp_size: args.tp_size,
                    cuda_graph: args.cuda_graph,
                    dump_graph_png: args.dump_graph_png.clone(),
                    offload,
                    no_prefix_cache: args.no_prefix_cache,
                    max_prefill_tokens: args
                        .max_prefill_tokens
                        .unwrap_or(openinfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS),
                    memory: openinfer_qwen3::Qwen3MemoryOptions::new(
                        args.gpu_memory_utilization,
                        kv_cache_memory_margin_bytes,
                        args.kv_page_size,
                    )
                    .validate()?,
                    lora,
                    decode_overlap: args.decode_overlap.resolve(args.decode_sm_pct),
                    batch_invariant: args.batch_invariant,
                    dflash_draft_model_path,
                    // KV block events are a Dynamo-backend concern; the plain
                    // server never publishes them.
                    enable_kv_events: false,
                },
            )
            .context("failed to start Qwen3 engine")?
        }
        #[cfg(feature = "qwen35")]
        ModelType::Qwen35 => openinfer_qwen35::launch_with_options_and_policy(
            &args.model_path,
            openinfer_qwen35::Qwen35LaunchOptions {
                device_ordinal: args.device_ordinal,
                tp_size: args.tp_size,
                cuda_graph: args.cuda_graph,
                max_batch: args.max_batch.unwrap_or(openinfer_qwen35::MAX_DECODE_BATCH),
                max_prefill_tokens: args
                    .max_prefill_tokens
                    .unwrap_or(openinfer_qwen35::DEFAULT_MAX_PREFILL_TOKENS),
            },
            args.qwen35_scheduler_policy.resolve(),
        )
        .context("failed to start Qwen3.5 engine")?,
    };

    Ok(handle)
}
