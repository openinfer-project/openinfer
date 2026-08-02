//! Disaggregated-prefill (native MTP P/D) protocol + the per-request KV
//! resolver. Storage orchestration lives in `openinfer-kv-store`
//! (`resolve_prefix` / `seal` / `retire` / keyed tail verbs); this module
//! keeps only what is GLM5.2 business: the handoff envelope, the anchor
//! plan, the KV shape contract, and the linear resolver that turns an
//! incoming request into a scheduler-ready intake — replacing the former
//! poll/park state machines (`HostRestoreState`, `NativePdState`). The
//! engine loop never blocks on any of this: resolution runs on the store's
//! runtime and completed intakes arrive over a channel.

use std::sync::Arc;

use anyhow::Context as _;
use openinfer_core::engine::GenerateRequest;
use openinfer_core::engine::KvPrefix;
use openinfer_kv_store::BlockPool;
use openinfer_kv_store::CacheScope;
use openinfer_kv_store::KvStore;
use openinfer_kv_store::RequestKv;
use openinfer_kv_store::ResolvePolicy;
use serde::Deserialize;

use super::PAGE;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct NativeMtpHandoff {
    pub(super) draft_tokens: [u32; crate::mtp::GLM52_MTP_DRAFTS],
    pub(super) committed_len: usize,
    pub(super) arena_count: usize,
    pub(super) tail_len: usize,
    pub(super) tail_key: Option<String>,
    pub(super) anchor_token_id: u32,
    /// Whether the anchor is client-visible; false only when it is an EOS
    /// consumed by P but suppressed from the response, so D must not replay it.
    pub(super) anchor_emitted: bool,
}

#[derive(Deserialize)]
struct NativeMtpEnvelope {
    version: u32,
    native_mtp: NativeMtpHandoff,
}

#[derive(Deserialize)]
struct OpenInferPdEnvelope {
    openinfer_pd: NativeMtpEnvelope,
}

pub(super) fn native_mtp_handoff(
    req: &GenerateRequest,
) -> anyhow::Result<Option<NativeMtpHandoff>> {
    let Some(value) = req.kv_transfer_params.clone() else {
        return Ok(None);
    };
    if value.get("openinfer_pd").is_none() {
        return Ok(None);
    }
    let envelope: OpenInferPdEnvelope =
        serde_json::from_value(value).context("invalid openinfer native-MTP P/D metadata")?;
    let version = envelope.openinfer_pd.version;
    anyhow::ensure!(
        version == 2,
        "unsupported openinfer P/D metadata version {version}"
    );
    let handoff = envelope.openinfer_pd.native_mtp;
    anyhow::ensure!(
        handoff.arena_count == 101,
        "native-MTP P/D requires 101 arenas, got {}",
        handoff.arena_count
    );
    Ok(Some(handoff))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeAnchorPlan {
    pub(super) token: u32,
    pub(super) replay_to_client: bool,
    pub(super) emitted_by_prefill: bool,
}

/// Resolve the v2 router contract without mutating the queued request.
///
/// vLLM Router forwards the original request unchanged, so D must append and
/// replay P's anchor. A manual v2 harness may already append the anchor and
/// combine P+D output itself; keep accepting that shape without replay.
pub(super) fn native_anchor_plan(
    req: &GenerateRequest,
    handoff: &NativeMtpHandoff,
) -> anyhow::Result<NativeAnchorPlan> {
    let token = handoff.anchor_token_id;
    let emitted_by_prefill = handoff.anchor_emitted;
    if req.prompt_tokens.len() == handoff.committed_len {
        return Ok(NativeAnchorPlan {
            token,
            replay_to_client: true,
            emitted_by_prefill,
        });
    }
    anyhow::ensure!(
        req.prompt_tokens.len() == handoff.committed_len + 1
            && req.prompt_tokens.last() == Some(&token),
        "native-MTP P/D v2 expects the original prompt or committed KV + anchor"
    );
    Ok(NativeAnchorPlan {
        token,
        replay_to_client: false,
        emitted_by_prefill,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeKvShape {
    pub(super) input_tokens: usize,
    pub(super) max_output_tokens: usize,
}

/// Exact kvbm geometry of a native-MTP D request. Router handoffs append P's
/// anchor to the logical input; manual handoffs already carry it and instead
/// need one extra internal output position because that anchor did not consume
/// the D request's client-visible output budget.
pub(super) fn native_kv_shape(
    req: &GenerateRequest,
    anchor_plan: NativeAnchorPlan,
) -> anyhow::Result<NativeKvShape> {
    let input_tokens = req
        .prompt_tokens
        .len()
        .checked_add(usize::from(anchor_plan.replay_to_client))
        .context("native-MTP P/D KV input length overflow")?;
    let anchor_counts_against_client_budget =
        anchor_plan.replay_to_client && anchor_plan.emitted_by_prefill;
    let max_output_tokens = req
        .max_tokens
        .checked_add(usize::from(!anchor_counts_against_client_budget))
        .context("native-MTP P/D KV output budget overflow")?;
    Ok(NativeKvShape {
        input_tokens,
        max_output_tokens,
    })
}

/// Keep the final committed page out of the lineage-hashed full-page prefix.
/// KV block hashes require a dangling token after a sealed page, so an
/// exactly page-aligned context still has a PAGE-token explicit tail.
pub(super) fn native_pd_tail_len(committed_len: usize) -> usize {
    committed_len
        .checked_sub(1)
        .map_or(0, |last| last % PAGE + 1)
}

/// A resolved intake, produced on the store runtime and drained by the
/// engine loop: the inbox holds only scheduler-ready requests.
pub(super) enum Resolved {
    /// Plain request: prefix resolution ran (or was skipped); admission does
    /// its normal match and drops the hold.
    Plain {
        req: GenerateRequest,
        prefix: KvPrefix,
    },
    /// Native-P/D request: the resolver built the COMPLETE `RequestKv`
    /// (full-page restore + tail install + anchor adoption) — admission only
    /// budgets and slots it.
    Native {
        req: GenerateRequest,
        kv: Box<RequestKv>,
        cached_tokens: usize,
        handoff: NativeMtpHandoff,
        plan: NativeAnchorPlan,
    },
    /// Resolution failed terminally (bad envelope, missing checkpoint past
    /// the deadline): admission answers with the standard rejection.
    Failed {
        req: GenerateRequest,
        message: String,
    },
}

impl Resolved {
    /// Surrender the request, dropping any resolution state (a built KV's
    /// blocks return via RAII; a prefix hold releases).
    pub(super) fn into_request(self) -> GenerateRequest {
        match self {
            Resolved::Plain { req, .. }
            | Resolved::Native { req, .. }
            | Resolved::Failed { req, .. } => req,
        }
    }
}

/// The decode side of the handoff, as one linear flow (formerly the
/// FullQuery -> FullLoad -> TailQuery -> TailLoad park machine):
/// full pages arrive via `resolve_prefix` in full-hit mode (registered into
/// the radix, so the fresh request's match reuses them), the tail page — a
/// partial page with no lineage hash — is fetched by its envelope key
/// straight into the request's own scheduled page, and the anchor converts
/// the final uncomputed token into normal dangling-token state.
pub(super) async fn native_pd_resolve(
    store: &KvStore,
    pool: &Arc<BlockPool>,
    rank: usize,
    req: &GenerateRequest,
    handoff: &NativeMtpHandoff,
) -> anyhow::Result<(RequestKv, usize)> {
    let plan = native_anchor_plan(req, handoff)?;
    anyhow::ensure!(
        handoff.tail_len == native_pd_tail_len(handoff.committed_len),
        "native-MTP P/D tail length {} disagrees with committed length {}",
        handoff.tail_len,
        handoff.committed_len
    );
    anyhow::ensure!(
        (handoff.tail_len == 0) == handoff.tail_key.is_none(),
        "native-MTP P/D tail key presence disagrees with tail length"
    );
    let cache_salt = super::native_mtp_cache_salt();
    let full_len = handoff.committed_len - handoff.tail_len;
    let req_id = req.request_id.as_deref().unwrap_or("native-pd");
    let t_start = std::time::Instant::now();

    // Full pages: all-or-nothing against the producer's checkpoint. The
    // store waits out registration lag and pool pressure under its deadline;
    // a short hit past it is terminal (decode cannot recompute the miss).
    let prompt_kv = &req.prompt_tokens[..handoff.committed_len];
    let prefix = store
        .resolve_prefix(
            rank,
            req_id,
            prompt_kv,
            CacheScope::default().cache_salt(cache_salt),
            ResolvePolicy::default().wait_for_full_hit(),
            &req.token_tx,
        )
        .await;
    let t_full = t_start.elapsed();
    anyhow::ensure!(
        prefix.hit_tokens() >= full_len,
        "full-page restore resolved {} of {} tokens before the deadline",
        prefix.hit_tokens(),
        full_len
    );

    let mut logical_prompt = req.prompt_tokens.clone();
    if plan.replay_to_client {
        logical_prompt.push(plan.token);
    }
    let logical_prompt_len = logical_prompt.len();
    let kv_shape = native_kv_shape(req, plan)?;
    anyhow::ensure!(
        logical_prompt_len == kv_shape.input_tokens,
        "native-MTP P/D KV input shape drift: built {logical_prompt_len}, planned {}",
        kv_shape.input_tokens
    );
    let mut kv = pool.new_request_with_cache_salt(
        logical_prompt,
        kv_shape.max_output_tokens,
        Some(cache_salt),
        None,
    );
    let mut cached_tokens = kv.match_and_add_prefix(pool)?;
    drop(prefix); // rematch done — the request itself holds the pages now
    anyhow::ensure!(
        cached_tokens >= full_len,
        "full-page rematch found {cached_tokens} of {full_len} resolved tokens"
    );

    if handoff.tail_len > 0 && cached_tokens < handoff.committed_len {
        let key_bytes = hex::decode(handoff.tail_key.as_deref().expect("validated tail key"))
            .context("native-MTP P/D tail key is not hex")?;
        let key: [u8; 16] = key_bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("native-MTP P/D tail key must be 16 bytes"))?;
        kv.schedule_prefill(handoff.tail_len, pool)
            .map_err(|err| anyhow::anyhow!("native P/D tail schedule: {err}"))?;
        let tail_page = *kv
            .step_page_indices(handoff.tail_len)
            .last()
            .expect("tail schedule owns one page");
        match store
            .resolve_keyed_block(rank, req_id, key, tail_page)
            .await
        {
            Ok(()) => {}
            Err(err) => {
                // The keyed load settles before returning, so reverting the
                // schedule here cannot race a DMA into the reverted page.
                let _ = kv.revert_schedule();
                return Err(err.context("native P/D tail install"));
            }
        }
        kv.apply_prefill_chunk(pool)?;
        cached_tokens += handoff.tail_len;
    }
    let t_total = t_start.elapsed();
    log::info!(
        "native P/D resolve timing: full_pages={}ms tail_and_match={}ms total={}ms \
         (committed_len={}, full_len={full_len})",
        t_full.as_millis(),
        (t_total - t_full).as_millis(),
        t_total.as_millis(),
        handoff.committed_len,
    );

    anyhow::ensure!(
        cached_tokens == handoff.committed_len && logical_prompt_len - kv.kv_position() == 1,
        "native-MTP P/D install incomplete: cached {cached_tokens} of {}, {} uncomputed",
        handoff.committed_len,
        logical_prompt_len - kv.kv_position(),
    );
    kv.adopt_external_prefill_anchor()
        .context("native-MTP P/D anchor adoption")?;
    Ok((kv, cached_tokens))
}
