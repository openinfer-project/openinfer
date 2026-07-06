//! DP8 lock-step continuous-batching scheduler: up to
//! `GLM52_MAX_BATCH_PER_RANK` requests per rank, each owning one slot of the
//! rank's decode batch. KV pages come from a per-rank [`BlockPool`]
//! (64-token pages, content-hashed blocks): admission reserves a request's
//! full-lifetime page count up front (honor-or-reject — a request that can
//! never fit is rejected, one that can't fit *now* stays queued), so decode
//! can never run out of pages mid-request, and released requests' sealed
//! blocks stay matchable as the prefix cache.
//!
//! Every global step ALL ranks run the full-model forward simultaneously with
//! the SAME batch bucket — ranks feed each active slot's *span* of next
//! tokens (mid-prefill slots batch up to a bucket of consecutive prompt
//! positions through one step; decode slots feed one row), idle slots feed a
//! padding row whose output is discarded. This satisfies the DeepEP contract
//! that every rank enters every MoE layer's dispatch/combine collective with
//! the agreed global row count. The bucket is the smallest member of
//! `GLM52_DECODE_BUCKETS` covering the hungriest rank's row demand, so the
//! fleet pays for prefill only while someone is prefilling and returns to the
//! cheap 1-row bucket for pure decode. Requests join and leave slots at step
//! boundaries (continuous batching) — admission is least-loaded rank first,
//! so decode-only fleets leave the 1-row bucket only past `GLM52_EP_RANKS`
//! concurrent requests.
//!
//! The per-request decisions (what to feed next, what a step's output means)
//! live in [`Glm52SlotState`] as pure data transitions, and the
//! admission/step-shape decisions in [`admission_target`] /
//! [`plan_step_shapes`] as pure functions over the occupancy and feed wants;
//! the coordinator is a thin shell that moves tokens between channels and the
//! rank workers.

use openinfer_core::engine::{FinishReason, GenerateRequest, TokenEvent, unix_now_s};
use openinfer_kv_cache::{BlockPool, RequestKv};
use openinfer_sample::{SamplingParams, mix_seed};
use tokio::sync::mpsc;

use crate::config::GLM52_VOCAB;
use crate::dspark::{GLM52_DSPARK_DRAFTS, accept_greedy};
use crate::model::{
    GLM52_DECODE_BUCKETS, GLM52_MAX_BATCH_PER_RANK, GLM52_MLA_TOPK_SHORT, GLM52_MODEL_LEN_ALIGN,
    Glm52StepKv, Glm52StepShape, glm52_pool_blocks, glm52_table_width,
};
use crate::runner::{Glm52RankWorker, Glm52RowSample, Glm52StepFlags};

/// The KV page size (== the FlashMLA page / index-K block / model-len
/// alignment — one 64 everywhere).
const PAGE: usize = GLM52_MODEL_LEN_ALIGN;

/// What a rank forwards this step. Idle rows feed the padding input; their
/// KV/index-cache writes land in the pool's reserved padding page, which no
/// request is ever assigned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52StepInput {
    pub(crate) token: u32,
    pub(crate) position: usize,
}

pub(crate) const GLM52_PADDING_STEP: Glm52StepInput = Glm52StepInput {
    token: 0,
    position: 0,
};

/// The consequence of one step's span of outputs for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Glm52StepOutcome {
    /// Mid-prefill: the model's outputs are discarded, keep feeding the prompt.
    Prefilling,
    /// Commit the span's agreed tokens: `committed` is the consumed run
    /// (what advances the request's KV bookkeeping — a suppressed EOS is its
    /// last entry), of which the leading `emit` tokens are sent to the
    /// client, then finish if `finish` is set. Plain decode commits exactly
    /// one token; a verify span commits the accepted draft prefix plus the
    /// model's correction or bonus token (1..=span tokens).
    Commit {
        committed: Vec<u32>,
        emit: usize,
        finish: Option<FinishReason>,
        /// Leading span rows whose tokens are now committed context for the
        /// draft lane: all rows of a prompt span; anchor + accepted drafts of
        /// a verify span (rejected rows' captured hidden is dead).
        context_rows: usize,
    },
}

/// One rank's active request as a pure state machine: `feed_want` /
/// `next_input_at` decide what the rank forwards (a span of consecutive
/// prompt positions mid-prefill, one decode row after), `advance_span` folds
/// the step's span of outputs in.
#[derive(Debug)]
pub(crate) struct Glm52SlotState {
    prompt: Vec<u32>,
    max_tokens: usize,
    ignore_eos: bool,
    /// Prompt tokens already fed to the model.
    fed: usize,
    /// Generated tokens (a suppressed EOS counts).
    completion: usize,
    /// The latest committed token; the next span's anchor once the prompt is
    /// fully fed.
    last_token: u32,
    /// The current spec-round proposal from the rank's draft lane, consumed
    /// (and cleared) by the next span. Empty = plain single-row decode — the
    /// drafter-off path is this same code with `drafts` never set.
    drafts: Vec<u32>,
    /// Accept telemetry across the request's verify rounds.
    spec: SpecStats,
}

/// Drafts fed per verify span: 3 drafts + anchor = a bucket-4 verify step.
/// A/B-measured on jz-38 (2026-07-04, docs/models/glm52/dspark-mtp.md): the
/// bucket-4 step costs ~32 ms vs bucket-8's ~46, and that cheaper round beats
/// span 8's extra accepted tail on EVERY tested prompt class — span 8 even
/// loses to plain decode on low-accept prose. The drafter still proposes 7;
/// the tail is simply not fed.
const GLM52_DSPARK_SPAN_DRAFTS: usize = 3;

/// Engine-level philox seed for unseeded non-greedy rows (the Kimi
/// convention: unseeded requests need no replay guarantee, so a fixed engine
/// seed suffices; per-request `seed` params replay through `mix_seed`).
const GLM52_SAMPLE_SEED: u64 = 42;

/// Accept histogram over a request's verify rounds (spans that actually fed
/// drafts; bonus-only single-row spans don't count).
#[derive(Debug, Default)]
struct SpecStats {
    rounds: u64,
    accepted_sum: u64,
    hist: [u64; GLM52_DSPARK_DRAFTS + 1],
}

impl Glm52SlotState {
    /// `cached_tokens` = the prefix-cache hit: those leading prompt tokens'
    /// KV is already resident in pool pages, so feeding starts past them
    /// (the matcher always leaves >= 1 prompt token uncached).
    pub(crate) fn new(
        prompt: Vec<u32>,
        max_tokens: usize,
        ignore_eos: bool,
        cached_tokens: usize,
    ) -> Self {
        debug_assert!(cached_tokens < prompt.len());
        Self {
            fed: cached_tokens,
            prompt,
            max_tokens,
            ignore_eos,
            completion: 0,
            last_token: 0,
            drafts: Vec::new(),
            spec: SpecStats::default(),
        }
    }

    pub(crate) fn completion_tokens(&self) -> usize {
        self.completion
    }

    /// Rows this request can usefully fill in one step: the whole remaining
    /// prompt while mid-prefill (the planner caps it to the bucket); the
    /// verify span (anchor + proposed drafts) in decode, capped so a round
    /// can never commit past `max_tokens` — which also keeps every fed
    /// position under the model-length cap, since `validate_request` pins
    /// `prompt + max_tokens - 1 <= max_model_len`.
    pub(crate) fn feed_want(&self) -> usize {
        if self.fed < self.prompt.len() {
            self.prompt.len() - self.fed
        } else {
            (1 + self.drafts.len()).min(self.max_tokens - self.completion)
        }
    }

    /// The `offset`-th row of this step's span: consecutive prompt positions
    /// while mid-prefill; the anchor (offset 0) then the draft prefix in
    /// decode. The planner may grant fewer rows than `feed_want` — the span
    /// is then a prefix (anchor + first drafts), and the un-fed drafts are
    /// discarded by `advance_span`.
    pub(crate) fn next_input_at(&self, offset: usize) -> Glm52StepInput {
        if self.fed < self.prompt.len() {
            debug_assert!(self.fed + offset < self.prompt.len());
            Glm52StepInput {
                token: self.prompt[self.fed + offset],
                position: self.fed + offset,
            }
        } else {
            debug_assert!(offset <= self.drafts.len());
            let token = if offset == 0 {
                self.last_token
            } else {
                self.drafts[offset - 1]
            };
            Glm52StepInput {
                token,
                position: self.prompt.len() + self.completion - 1 + offset,
            }
        }
    }

    /// The span row whose output `advance_span` will commit, for the sampler
    /// to overwrite: the prompt-completing span's last row (its output is the
    /// first generated token) or the plain decode row. `None` while the span
    /// is still mid-prompt (outputs discarded). Only meaningful for
    /// non-greedy requests, which never carry drafts (DSpark verify is
    /// greedy-only), so a decode span is always the single anchor row.
    pub(crate) fn sampling_row(&self, span_rows: usize) -> Option<usize> {
        debug_assert!(span_rows > 0);
        if self.fed < self.prompt.len() {
            (self.fed + span_rows == self.prompt.len()).then(|| span_rows - 1)
        } else {
            // Sampling a verify span would feed a sampled token into
            // `accept_greedy` — whoever lifts the DSpark greedy-only gate
            // must build rejection sampling first, not trip this in release.
            assert!(
                self.drafts.is_empty(),
                "sampling a DSpark verify span is unsupported (greedy-only gate breached)"
            );
            Some(0)
        }
    }

    /// The next spec round's anchor once the request is decoding: the latest
    /// committed token and the position it will be fed at. `None` mid-prefill
    /// (no token to extend yet).
    pub(crate) fn decode_anchor(&self) -> Option<(u32, usize)> {
        (self.fed >= self.prompt.len() && self.completion > 0)
            .then(|| (self.last_token, self.prompt.len() + self.completion - 1))
    }

    /// Whether a fresh draft proposal is worth requesting: decoding, and at
    /// least two tokens of budget left (a one-token tail can only ever commit
    /// the anchor's own output — a plain row).
    pub(crate) fn wants_drafts(&self) -> bool {
        self.fed >= self.prompt.len() && self.completion + 1 < self.max_tokens
    }

    /// Install the draft lane's proposal for the next verify span, truncated
    /// to [`GLM52_DSPARK_SPAN_DRAFTS`].
    pub(crate) fn set_drafts(&mut self, mut drafts: Vec<u32>) {
        drafts.truncate(GLM52_DSPARK_SPAN_DRAFTS);
        self.drafts = drafts;
    }

    /// Fold one step's span of outputs in.
    ///
    /// Mid-prompt rows' outputs are discarded; the row that fed the LAST
    /// prompt token yields the first generated token. In decode the span is
    /// a verify: `outputs[k]` is the target's greedy token after span row
    /// `k` (anchor + fed draft prefix), and [`accept_greedy`] commits the
    /// agreed prefix plus one model token. The plain single-row step is the
    /// zero-draft special case of the same rule.
    pub(crate) fn advance_span(
        &mut self,
        outputs: &[u32],
        eos_token_ids: &[u32],
    ) -> Glm52StepOutcome {
        debug_assert!(!outputs.is_empty());
        let (committed, context_rows) = if self.fed < self.prompt.len() {
            debug_assert!(self.fed + outputs.len() <= self.prompt.len());
            self.fed += outputs.len();
            if self.fed < self.prompt.len() {
                return Glm52StepOutcome::Prefilling;
            }
            // Boundary: every span row is committed prompt context, and the
            // last row's output is the first generated token.
            let output = *outputs.last().expect("span outputs are non-empty");
            (vec![output], outputs.len())
        } else {
            let drafts_fed = outputs.len() - 1;
            debug_assert!(drafts_fed <= self.drafts.len());
            let committed = accept_greedy(&self.drafts[..drafts_fed], outputs);
            if drafts_fed > 0 {
                self.spec.record(committed.len() - 1);
            }
            let context_rows = committed.len();
            (committed, context_rows)
        };
        self.drafts.clear();

        let mut committed = committed;
        let mut emit = 0usize;
        let mut finish = None;
        for &token in &committed {
            self.completion += 1;
            if !self.ignore_eos && eos_token_ids.contains(&token) {
                finish = Some(FinishReason::Stop);
                break;
            }
            emit += 1;
            self.last_token = token;
            if self.completion >= self.max_tokens {
                finish = Some(FinishReason::Length);
                break;
            }
        }
        // Truncate to the consumed run (a suppressed EOS is consumed but not
        // emitted) so the caller's KV bookkeeping advances by exactly the
        // tokens this state accounted for.
        let consumed = emit + usize::from(matches!(finish, Some(FinishReason::Stop)));
        committed.truncate(consumed);
        Glm52StepOutcome::Commit {
            committed,
            emit,
            finish,
            context_rows,
        }
    }

    /// Log the request's accept telemetry when it leaves its slot (only when
    /// it ran verify rounds — plain-decode requests stay silent).
    pub(crate) fn log_spec_stats(&self, rank: usize, slot: usize) {
        let stats = &self.spec;
        if stats.rounds == 0 {
            return;
        }
        let mean_accepted = stats.accepted_sum as f64 / stats.rounds as f64;
        log::info!(
            "GLM5.2 dspark: rank={rank} slot={slot} rounds={} mean_accepted_drafts={mean_accepted:.3} \
             mean_accepted_incl_bonus={:.3} hist={:?}",
            stats.rounds,
            mean_accepted + 1.0,
            stats.hist,
        );
    }
}

impl SpecStats {
    fn record(&mut self, accepted_drafts: usize) {
        self.rounds += 1;
        self.accepted_sum += accepted_drafts as u64;
        self.hist[accepted_drafts.min(GLM52_DSPARK_DRAFTS)] += 1;
    }
}

pub(crate) fn validate_request(
    req: &GenerateRequest,
    max_model_len: usize,
    dspark_enabled: bool,
) -> Result<(), String> {
    if req.prompt_tokens.is_empty() {
        return Err("GLM5.2 requires a non-empty prompt".to_owned());
    }
    if req.max_tokens == 0 {
        return Err("GLM5.2 requires max_tokens > 0".to_owned());
    }
    // Highest position any forward step can touch: the (max_tokens-1)-th
    // generated token is fed at position prompt+max_tokens-2, so requiring
    // prompt+max_tokens-1 <= cap keeps every step strictly below the cap.
    let last_position = req.prompt_tokens.len() + req.max_tokens - 1;
    if last_position > max_model_len {
        return Err(format!(
            "GLM5.2 context cap: prompt {} + max_tokens {} exceeds max_model_len {max_model_len}",
            req.prompt_tokens.len(),
            req.max_tokens
        ));
    }
    // Plain decode samples per-row; the DSpark verify span is greedy-only
    // ([`accept_greedy`] compares the target's argmax against the drafts —
    // rejection sampling is future work).
    if dspark_enabled && !req.params.is_greedy() {
        return Err(
            "GLM5.2 with the DSpark drafter supports greedy sampling only (temperature 0)"
                .to_owned(),
        );
    }
    // Mirror the sampler kernel's parameter ensures HERE: past intake a bad
    // value only surfaces as a failed step, and a failed step tears the whole
    // EP8 engine down (`fail_step`) — user input must be rejected at the
    // door, never inside a collective.
    if !req.params.is_greedy() {
        let p = &req.params;
        if !p.temperature.is_finite() {
            return Err(format!(
                "GLM5.2 sampling requires a finite temperature, got {}",
                p.temperature
            ));
        }
        if !(p.top_p > 0.0 && p.top_p <= 1.0) {
            return Err(format!(
                "GLM5.2 sampling requires top_p in (0, 1], got {}",
                p.top_p
            ));
        }
        if !(p.min_p.is_finite() && (0.0..1.0).contains(&p.min_p)) {
            return Err(format!(
                "GLM5.2 sampling requires min_p in [0, 1), got {}",
                p.min_p
            ));
        }
    }
    if req.logprobs > 0 || req.echo {
        return Err("GLM5.2 bring-up does not support logprobs/echo".to_owned());
    }
    if req.lora_adapter.is_some() {
        return Err("GLM5.2 does not support LoRA adapters".to_owned());
    }
    Ok(())
}

struct ActiveRequest {
    req: GenerateRequest,
    state: Glm52SlotState,
    /// The request's page assignments in the rank's pool. Block RAII: blocks
    /// return to the pool (registered ones as matchable prefix-cache entries)
    /// when this drops or `release()`s.
    kv: RequestKv,
}

/// Per-rank slot occupancy: `slots[rank][slot]`.
type RankSlots = [Option<ActiveRequest>; GLM52_MAX_BATCH_PER_RANK];

/// Pool pages a request draws over its whole lifetime, reserved at
/// admission. One more token than the last KV-written position: kvbm appends
/// the final generated token to the sequence and provisions its page even
/// though its KV is never written (the dangling-token contract — the same
/// off-by-one Kimi's admission had to learn empirically).
fn lifetime_blocks(prompt_tokens: usize, max_tokens: usize) -> usize {
    (prompt_tokens + max_tokens).div_ceil(PAGE)
}

/// Where the next queued request goes: among the ranks with a free slot AND
/// enough unreserved pool pages for the request's full lifetime, the
/// least-loaded one (ties → lowest rank id), its lowest free slot. `None`
/// when no rank can take it — the queue holds (a request that fits the pool
/// geometry always fits an EMPTY rank, so FCFS deferral never livelocks).
/// Least-loaded-first keeps occupancy balanced, which keeps the fleet in the
/// cheap 1-row bucket until concurrency exceeds the rank count.
///
/// `committed[rank]` = Σ active requests' [`lifetime_blocks`];
/// `usable[rank]` = pool blocks minus the reserved padding page. The
/// reservation is conservative: prefix-cache hits share pages between
/// requests, but each holder reserves them in full — over-reserving can only
/// defer admission, never strand a decode.
fn admission_target(
    occupied: &[[bool; GLM52_MAX_BATCH_PER_RANK]],
    committed: &[usize],
    usable: &[usize],
    need_blocks: usize,
) -> Option<(usize, usize)> {
    let (rank, row) = occupied
        .iter()
        .enumerate()
        .filter(|(rank, row)| {
            committed[*rank] + need_blocks <= usable[*rank] && row.iter().any(|&o| !o)
        })
        .min_by_key(|(rank, row)| (row.iter().filter(|&&o| o).count(), *rank))?;
    let slot = row.iter().position(|&o| !o)?;
    Some((rank, slot))
}

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
fn plan_step_shapes(wants: &[[usize; GLM52_MAX_BATCH_PER_RANK]]) -> Vec<Glm52StepShape> {
    let hungriest = wants
        .iter()
        .map(|row| row.iter().sum::<usize>().min(GLM52_MAX_BATCH_PER_RANK))
        .max()
        .unwrap_or(0);
    let bucket = *GLM52_DECODE_BUCKETS
        .iter()
        .find(|&&rows| rows >= hungriest.max(1))
        .expect("the largest bucket covers every demand by construction");
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
fn launch_ahead_flags(
    shapes: &[Glm52StepShape],
    leased_shapes: Option<&[Glm52StepShape]>,
    slots_changed: bool,
    pending_empty: bool,
    dspark_enabled: bool,
    slots: &[RankSlots],
    max_model_len: usize,
) -> Glm52StepFlags {
    let consume = !slots_changed && leased_shapes == Some(shapes);
    let lease = pending_empty
        && !dspark_enabled
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
/// contiguous per-slot runs and mark each non-greedy slot's committed row
/// (see [`Glm52SlotState::sampling_row`]) with its request params and
/// request-local decode step. Rows come out strictly ascending — the runs
/// are disjoint and walked in order — which `sample_rows_into` re-checks.
fn collect_sampling_rows(shape: &Glm52StepShape, rank_slots: &RankSlots) -> Vec<Glm52RowSample> {
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
            && let Some(offset) = active.state.sampling_row(end - row)
        {
            sampling.push(Glm52RowSample {
                row: row + offset,
                params: active.req.params,
                step: active.state.completion_tokens() as u64,
            });
        }
        row = end;
    }
    sampling
}

fn feed_wants(slots: &[RankSlots]) -> Vec<[usize; GLM52_MAX_BATCH_PER_RANK]> {
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

fn occupancy(slots: &[RankSlots]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
    slots
        .iter()
        .map(|rank_slots| std::array::from_fn(|slot| rank_slots[slot].is_some()))
        .collect()
}

/// What one slot's span asked kvbm for this step — decides which `apply_*`
/// commits the outputs (schedule and apply must pair exactly).
#[derive(Clone, Copy, Debug)]
enum SpanKind {
    /// Prompt span that does NOT finish the prompt: KV advances, no token.
    PrefillChunk,
    /// Prompt span whose last row feeds the final prompt token: its output
    /// is the first generated token.
    PrefillBoundary,
    /// Single decode row (the zero-draft case).
    Decode,
    /// Verify span: anchor + fed drafts, committing the accepted prefix.
    Speculative,
}

/// The all-padding-rows step KV: every page-table entry is the pool's
/// padding page, every write slot points into it (positions are `< PAGE` by
/// construction for padding rows — pre-capture probes position
/// [`GLM52_MLA_TOPK_SHORT`], a PAGE multiple, and serving pads sit at 0).
fn padding_step_kv(
    bucket: usize,
    table_width: usize,
    padding_page: i32,
    inputs: &[(u32, usize); GLM52_MAX_BATCH_PER_RANK],
) -> Glm52StepKv {
    let pages = vec![padding_page; bucket * table_width].into_boxed_slice();
    let mut slot_mapping = [0i64; GLM52_MAX_BATCH_PER_RANK];
    for (row, slot) in slot_mapping.iter_mut().enumerate().take(bucket) {
        *slot = padding_page as i64 * PAGE as i64 + (inputs[row].1 % PAGE) as i64;
    }
    Glm52StepKv {
        pages,
        slot_mapping,
    }
}

/// DP8 coordinator: admits up to `GLM52_MAX_BATCH_PER_RANK` requests per rank
/// (least-loaded rank with pool budget first) and drives all ranks in
/// lock-step. Consumes the workers; returns when the submit channel closes
/// or a step fails (the EP8 collective group cannot recover from a failed
/// step — see the teardown comment below).
pub(crate) fn run_dp8_coordinator(
    mut submit_rx: mpsc::UnboundedReceiver<GenerateRequest>,
    workers: Vec<Glm52RankWorker>,
    eos_token_ids: &[u32],
    dspark_enabled: bool,
    max_model_len: usize,
    no_prefix_cache: bool,
) {
    // One KV page pool per rank: pool block ids index the rank's per-layer
    // MLA and index-K arenas directly (the arenas were built for
    // `glm52_pool_blocks` blocks). Block 0-equivalent is the reserved
    // padding page.
    let pools: Vec<BlockPool> = match workers
        .iter()
        .map(|_| BlockPool::new(PAGE, glm52_pool_blocks(max_model_len)))
        .collect::<anyhow::Result<Vec<_>>>()
    {
        Ok(pools) => pools,
        Err(err) => {
            log::error!("GLM5.2 KV pool construction failed: {err:#}");
            for worker in &workers {
                let _ = worker.request_shutdown();
            }
            return;
        }
    };
    let table_width = glm52_table_width(max_model_len);
    // Pool pages available to requests per rank (total minus the padding
    // page) — constant for the engine's lifetime.
    let usable_blocks: Vec<usize> = pools.iter().map(|pool| pool.total_blocks() - 1).collect();
    // The DSpark draft lane asserts every anchor position equals its
    // committed + pending context rows — a skipped (cache-hit) prefix never
    // produces the aux-hidden captures the draft consumes, so prefix
    // matching is off while the drafter is on. Speculative decoding and
    // prefix caching are mutually exclusive for now (the qwen3 offload path
    // draws the same line). `--no-prefix-cache` is the explicit kill switch.
    let prefix_cache_enabled = !dspark_enabled && !no_prefix_cache;
    if dspark_enabled && !no_prefix_cache {
        log::info!("GLM5.2 prefix cache disabled: the DSpark drafter is on");
    }
    let mut slots: Vec<RankSlots> = workers
        .iter()
        .map(|_| std::array::from_fn(|_| None))
        .collect();
    // Slot draft states to clear on the next draft round (request left the
    // slot, or a new one was admitted into it). Flushed with each step's
    // Draft commands; the handler is idempotent, so duplicates are harmless.
    let mut pending_resets: Vec<Vec<usize>> = workers.iter().map(|_| Vec::new()).collect();
    let mut pending = std::collections::VecDeque::<GenerateRequest>::new();
    let mut channel_open = true;
    let all_idle = |slots: &[RankSlots]| {
        slots
            .iter()
            .all(|rank_slots| rank_slots.iter().all(Option::is_none))
    };

    // Pre-capture every whole-step graph (bucket × attention tier) while the
    // ranks are idle and trivially in lock-step. Launch-ahead speculation
    // requires captured-ness to be UNIFORM across ranks — a lazily capturing
    // rank would skip the speculative replay the others enqueued and desync
    // the collectives — and pre-capturing also removes the old mid-serving
    // capture stall. Row 0 at position GLM52_MLA_TOPK_SHORT lifts the step
    // into the full tier; every row is a padding write into the pool's
    // padding page.
    for &bucket in &GLM52_DECODE_BUCKETS {
        for full_tier in [false, true] {
            let mut shape = Glm52StepShape {
                bucket,
                slots: [0; GLM52_MAX_BATCH_PER_RANK],
                active_rows: 0,
            };
            for (slot, dst) in shape.slots.iter_mut().enumerate().take(bucket) {
                *dst = slot as u8;
            }
            let mut inputs =
                [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_BATCH_PER_RANK];
            if full_tier {
                inputs[0] = (GLM52_PADDING_STEP.token, GLM52_MLA_TOPK_SHORT);
            }
            let responses = match workers
                .iter()
                .zip(&pools)
                .map(|(worker, pool)| {
                    let kv = padding_step_kv(bucket, table_width, pool.padding_block_id(), &inputs);
                    worker.step_async(inputs, shape, kv, Glm52StepFlags::plain(), Vec::new(), 0)
                })
                .collect::<anyhow::Result<Vec<_>>>()
            {
                Ok(responses) => responses,
                Err(err) => {
                    log::error!("GLM5.2 graph pre-capture failed to submit: {err:#}");
                    // The DeepEP contexts already exist: broadcast Shutdown
                    // before the workers' sequential Drop joins them (the
                    // same collective-teardown contract as the exit path).
                    for worker in &workers {
                        let _ = worker.request_shutdown();
                    }
                    return;
                }
            };
            for (rank, resp) in responses.into_iter().enumerate() {
                let result = resp
                    .recv()
                    .map_err(|_| anyhow::anyhow!("rank dropped its pre-capture response"))
                    .and_then(|r| r);
                if let Err(err) = result {
                    log::error!(
                        "GLM5.2 graph pre-capture (bucket {bucket}, full_tier {full_tier}) \
                         failed on rank {rank}: {err:#}"
                    );
                    for worker in &workers {
                        let _ = worker.request_shutdown();
                    }
                    return;
                }
            }
        }
    }
    log::info!(
        "GLM5.2 whole-step graphs pre-captured: {} buckets x 2 tiers",
        GLM52_DECODE_BUCKETS.len()
    );

    // The step shapes that carried the last launch-ahead lease, and whether
    // any slot changed hands since it was granted — together they decide
    // whether this step consumes the speculation (must be all-ranks-or-none,
    // so the decision lives here, on global data, not on the ranks).
    let mut leased_shapes: Option<Vec<Glm52StepShape>> = None;
    let mut slots_changed = false;
    // Global step counter driving the non-greedy rows' philox seeds: a fresh
    // well-mixed seed per (step, rank), so no two ranks — whose row indices
    // collide — ever share a philox stream.
    let mut sample_step: u64 = 0;

    'serve: loop {
        // Intake: block when fully idle, otherwise drain what's queued.
        if channel_open && all_idle(&slots) && pending.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => intake(req, &mut pending, max_model_len, dspark_enabled),
                None => channel_open = false,
            }
        }
        while channel_open {
            match submit_rx.try_recv() {
                Ok(req) => intake(req, &mut pending, max_model_len, dspark_enabled),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => channel_open = false,
            }
        }
        if !channel_open && all_idle(&slots) && pending.is_empty() {
            break;
        }

        // Admission: fill free slots from the queue, least-loaded rank (with
        // pool budget for the request's full lifetime) first. New requests
        // join the lock-step at the next step boundary (their prefill rides
        // decode alongside everyone else's rows).
        while let Some(front) = pending.front() {
            let need_blocks = lifetime_blocks(front.prompt_tokens.len(), front.max_tokens);
            let committed: Vec<usize> = slots
                .iter()
                .map(|rank_slots| {
                    rank_slots
                        .iter()
                        .flatten()
                        .map(|active| {
                            lifetime_blocks(active.req.prompt_tokens.len(), active.req.max_tokens)
                        })
                        .sum()
                })
                .collect();
            let Some((rank, slot)) =
                admission_target(&occupancy(&slots), &committed, &usable_blocks, need_blocks)
            else {
                break;
            };
            let req = pending.pop_front().expect("checked non-empty");
            // The client left while the request sat in the queue — admitting
            // it would burn a slot (and whole global steps) on a dead sink.
            if req.token_tx.is_closed() {
                continue;
            }
            let mut kv = pools[rank].new_request(req.prompt_tokens.clone(), req.max_tokens, None);
            let cached_tokens = if prefix_cache_enabled {
                match kv.match_and_add_prefix(&pools[rank]) {
                    Ok(cached) => cached,
                    Err(err) => {
                        // A fresh request failing to match is a kvbm state
                        // invariant break — crash early, don't serve on a
                        // pool whose bookkeeping already lied once. This
                        // request is already out of `pending` and never
                        // reaches a slot, so fail it explicitly (fail_step
                        // and the shutdown sweep can't see it).
                        let err = err.context("GLM5.2 prefix match at admission");
                        let _ = req.token_tx.send(TokenEvent::Error {
                            message: format!("{err:#}"),
                            prompt_tokens: req.prompt_tokens.len(),
                            completion_tokens: 0,
                        });
                        fail_step(&mut slots, &err);
                        break 'serve;
                    }
                }
            } else {
                0
            };
            let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
            let _ = req.token_tx.send(TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s: unix_now_s(),
                prompt_tokens: req.prompt_tokens.len(),
                cached_tokens,
            });
            let state = Glm52SlotState::new(
                req.prompt_tokens.clone(),
                req.max_tokens,
                req.params.ignore_eos,
                cached_tokens,
            );
            if dspark_enabled {
                pending_resets[rank].push(slot);
            }
            slots[rank][slot] = Some(ActiveRequest { req, state, kv });
            slots_changed = true;
        }
        if all_idle(&slots) {
            continue;
        }

        // One lock-step step: every rank forwards the SAME bucket — each
        // active slot's span of consecutive next tokens, padding rows on the
        // free slots — and all responses are joined before any output is
        // interpreted.
        let shapes = plan_step_shapes(&feed_wants(&slots));
        let flags = launch_ahead_flags(
            &shapes,
            leased_shapes.as_deref(),
            slots_changed,
            pending.is_empty(),
            dspark_enabled,
            &slots,
            max_model_len,
        );
        leased_shapes = flags.lease.then(|| shapes.clone());
        slots_changed = false;
        sample_step += 1;
        // Per-rank submit: schedule each active span's KV (full-lifetime
        // reservation makes every schedule succeed — a failure is an
        // accounting bug and fails the step), build the row inputs, page
        // rows and write slots, collect the step's sampling rows, and fire
        // the step. `span_kinds[rank][slot]` records what was scheduled so
        // the output walk applies the exact pairing.
        let mut span_kinds: Vec<[Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK]> = workers
            .iter()
            .map(|_| [None; GLM52_MAX_BATCH_PER_RANK])
            .collect();
        let mut responses = Vec::with_capacity(workers.len());
        let mut submit_err: Option<anyhow::Error> = None;
        'submit: for (rank, ((rank_slots, worker), shape)) in
            slots.iter_mut().zip(&workers).zip(&shapes).enumerate()
        {
            let pool = &pools[rank];
            let padding_page = pool.padding_block_id();
            let sampling = collect_sampling_rows(shape, rank_slots);
            let seed = mix_seed(mix_seed(GLM52_SAMPLE_SEED, sample_step), rank as u64);
            let mut inputs =
                [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_BATCH_PER_RANK];
            // A consumed speculation replays with device-advanced inputs and
            // never reads the step KV — skip building the page rows (the
            // whole point of launch-ahead is keeping this host path off the
            // hot step boundary). KV *scheduling* still runs: kvbm's
            // bookkeeping must advance every step.
            let mut pages = if flags.consume {
                Vec::new()
            } else {
                vec![padding_page; shape.bucket * table_width]
            };
            let mut slot_mapping = [padding_page as i64 * PAGE as i64; GLM52_MAX_BATCH_PER_RANK];
            // Walk the shape's contiguous per-slot runs.
            let mut row = 0usize;
            while row < shape.bucket {
                let slot_id = shape.slots[row] as usize;
                let mut end = row + 1;
                while end < shape.bucket && shape.slots[end] as usize == slot_id {
                    end += 1;
                }
                let span = end - row;
                let Some(active) = rank_slots[slot_id].as_mut() else {
                    // Padding rows keep the padding-page defaults.
                    row = end;
                    continue;
                };
                for (offset, r) in (row..end).enumerate() {
                    let step = active.state.next_input_at(offset);
                    inputs[r] = (step.token, step.position);
                }
                // The span must extend kvbm's view exactly: its first row's
                // position is the next KV slot to write. Drift between the
                // slot state's position math and the pool's bookkeeping
                // writes KV into the wrong page — fail the step instead.
                if inputs[row].1 != active.kv.kv_position() {
                    submit_err = Some(anyhow::anyhow!(
                        "GLM5.2 rank {rank} slot {slot_id} span starts at position {} but the \
                         KV pool is at {}",
                        inputs[row].1,
                        active.kv.kv_position()
                    ));
                    break 'submit;
                }
                let mid_prefill = active.state.fed < active.state.prompt.len();
                let (kind, scheduled) = if mid_prefill {
                    let kind = if active.state.fed + span == active.state.prompt.len() {
                        SpanKind::PrefillBoundary
                    } else {
                        SpanKind::PrefillChunk
                    };
                    (kind, active.kv.schedule_prefill(span, pool))
                } else if span == 1 {
                    (SpanKind::Decode, active.kv.schedule_decode(pool))
                } else {
                    (
                        SpanKind::Speculative,
                        active.kv.schedule_speculative(span, pool),
                    )
                };
                if let Err(err) = scheduled {
                    submit_err = Some(anyhow::anyhow!(
                        "GLM5.2 rank {rank} slot {slot_id} violated its full-lifetime KV \
                         reservation ({kind:?}, span {span}): {err}"
                    ));
                    break 'submit;
                }
                span_kinds[rank][slot_id] = Some(kind);
                if !flags.consume {
                    let row_pages = active.kv.step_page_indices(span);
                    for r in row..end {
                        pages[r * table_width..r * table_width + row_pages.len()]
                            .copy_from_slice(&row_pages);
                        let position = inputs[r].1;
                        slot_mapping[r] = row_pages[position / PAGE] as i64 * PAGE as i64
                            + (position % PAGE) as i64;
                    }
                }
                row = end;
            }
            let kv = Glm52StepKv {
                pages: pages.into_boxed_slice(),
                slot_mapping,
            };
            match worker.step_async(inputs, *shape, kv, flags, sampling, seed) {
                Ok(rx) => responses.push(rx),
                Err(err) => {
                    submit_err = Some(err);
                    break 'submit;
                }
            }
        }
        if let Some(err) = submit_err {
            fail_step(&mut slots, &err);
            break 'serve;
        }
        // Join ALL ranks before failing: the rank the coordinator happens to
        // recv first often reports the ~100 s DeepEP device-timeout trap, not
        // the root cause — the rank that actually failed answers later. Log
        // every error so the root cause has a landing spot, then tear down on
        // the first one in rank order.
        let mut outputs = Vec::with_capacity(responses.len());
        let mut step_err: Option<anyhow::Error> = None;
        for (rank, resp) in responses.into_iter().enumerate() {
            let result = resp
                .recv()
                .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its step response"));
            match result {
                Ok(Ok(step_tokens)) => outputs.push(step_tokens),
                Ok(Err(err)) | Err(err) => {
                    let err = err.context(format!("GLM5.2 rank {rank} step"));
                    log::error!("GLM5.2 rank {rank} step failed: {err:#}");
                    step_err.get_or_insert(err);
                    outputs.push([0; GLM52_MAX_BATCH_PER_RANK]);
                }
            }
        }
        if let Some(err) = step_err {
            fail_step(&mut slots, &err);
            break 'serve;
        }

        let mut rank_appends: Vec<Vec<(usize, usize)>> =
            workers.iter().map(|_| Vec::new()).collect();
        let mut rank_proposals: Vec<Vec<(usize, u32, usize)>> =
            workers.iter().map(|_| Vec::new()).collect();
        let mut walk_err: Option<anyhow::Error> = None;
        'walk: for (rank, ((rank_slots, rank_outputs), shape)) in
            slots.iter_mut().zip(outputs).zip(&shapes).enumerate()
        {
            // Walk the shape's contiguous per-slot runs; each active slot
            // folds its whole span of row outputs in at once.
            let mut row = 0usize;
            while row < shape.bucket {
                let slot_id = shape.slots[row] as usize;
                let mut end = row + 1;
                while end < shape.bucket && shape.slots[end] as usize == slot_id {
                    end += 1;
                }
                let span_rows = row..end;
                let span_outputs = &rank_outputs[span_rows.clone()];
                row = end;
                let slot = &mut rank_slots[slot_id];
                let Some(active) = slot.as_mut() else {
                    continue;
                };
                let prompt_tokens = active.req.prompt_tokens.len();
                let outcome = active.state.advance_span(span_outputs, eos_token_ids);
                // Commit the span's KV bookkeeping under the exact kind the
                // submit phase scheduled — a mispairing is a coordinator bug
                // and fails the step.
                let pool = &pools[rank];
                let applied = match (&outcome, span_kinds[rank][slot_id]) {
                    (Glm52StepOutcome::Prefilling, Some(SpanKind::PrefillChunk)) => {
                        active.kv.apply_prefill_chunk(pool)
                    }
                    (
                        Glm52StepOutcome::Commit { committed, .. },
                        Some(SpanKind::PrefillBoundary),
                    ) => active.kv.apply_prefill(committed[0], pool),
                    (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::Decode)) => {
                        active.kv.apply_decode(committed[0], pool).map(|_| ())
                    }
                    (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::Speculative)) => {
                        active.kv.apply_speculative(committed, pool).map(|_| ())
                    }
                    (outcome, kind) => Err(anyhow::anyhow!(
                        "GLM5.2 rank {rank} slot {slot_id} outcome {outcome:?} does not pair \
                         with scheduled span kind {kind:?}"
                    )),
                };
                if let Err(err) = applied {
                    walk_err =
                        Some(err.context(format!("GLM5.2 rank {rank} slot {slot_id} KV apply")));
                    break 'walk;
                }
                let (freed, context_rows) = match outcome {
                    Glm52StepOutcome::Prefilling => {
                        // Prefill never sends, so a disconnect is only
                        // visible through the sink probe — without it a
                        // long prompt zombies the slot until prefill
                        // completes. Every prompt row is committed
                        // context.
                        (active.req.token_tx.is_closed(), span_outputs.len())
                    }
                    Glm52StepOutcome::Commit {
                        committed,
                        emit,
                        finish,
                        context_rows,
                    } => {
                        // A dropped receiver (client disconnect) frees the
                        // slot; its pool pages release with the request
                        // (sealed blocks stay matchable as prefix cache).
                        let mut freed = false;
                        for &token in &committed[..emit] {
                            if active
                                .req
                                .token_tx
                                .send(TokenEvent::Token {
                                    id: token,
                                    logprob: None,
                                })
                                .is_err()
                            {
                                freed = true;
                                break;
                            }
                        }
                        if let Some(finish_reason) = finish
                            && !freed
                        {
                            let _ = active.req.token_tx.send(TokenEvent::Finished {
                                finish_reason,
                                prompt_tokens,
                                completion_tokens: active.state.completion_tokens(),
                            });
                            freed = true;
                        }
                        (freed, context_rows)
                    }
                };
                if freed {
                    active.state.log_spec_stats(rank, slot_id);
                    if let Err(err) = active.kv.release() {
                        // Blocks still return via assignment RAII when the
                        // slot drops — the explicit release only failed to
                        // run from a clean Idle state.
                        log::warn!(
                            "GLM5.2 rank {rank} slot {slot_id} KV release failed \
                             (blocks return via RAII): {err:#}"
                        );
                    }
                    if dspark_enabled {
                        pending_resets[rank].push(slot_id);
                    }
                    *slot = None;
                    slots_changed = true;
                } else if dspark_enabled {
                    // Committed rows' captured hidden feeds the draft
                    // context; then re-propose from the new anchor.
                    rank_appends[rank].extend(span_rows.take(context_rows).map(|r| (r, slot_id)));
                    if active.state.wants_drafts()
                        && let Some((anchor, anchor_pos)) = active.state.decode_anchor()
                    {
                        rank_proposals[rank].push((slot_id, anchor, anchor_pos));
                    }
                }
            }
        }
        if let Some(err) = walk_err {
            fail_step(&mut slots, &err);
            break 'serve;
        }

        // Draft round (rank-local, no collectives): resets, context appends
        // from THIS step's capture buffer, and new proposals for the next
        // verify span. FIFO per-rank channels order it before the next step;
        // the blocking join keeps the round cadence (draft sits between
        // verify steps, ~2 ms against a 22-46 ms step).
        if dspark_enabled {
            let mut draft_joins = Vec::new();
            for (rank, worker) in workers.iter().enumerate() {
                let resets = std::mem::take(&mut pending_resets[rank]);
                let appends = std::mem::take(&mut rank_appends[rank]);
                let proposals = std::mem::take(&mut rank_proposals[rank]);
                if resets.is_empty() && appends.is_empty() && proposals.is_empty() {
                    continue;
                }
                let proposal_slots: Vec<usize> =
                    proposals.iter().map(|&(slot, _, _)| slot).collect();
                match worker.draft_async(shapes[rank].bucket, resets, appends, proposals) {
                    Ok(rx) => draft_joins.push((rank, proposal_slots, rx)),
                    Err(err) => {
                        fail_step(&mut slots, &err);
                        break 'serve;
                    }
                }
            }
            for (rank, proposal_slots, rx) in draft_joins {
                let result = rx
                    .recv()
                    .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its draft response"))
                    .and_then(|r| r);
                let spans = match result {
                    Ok(spans) => spans,
                    Err(err) => {
                        // A draft failure is rank-local, but it means the
                        // drafter's invariants broke — crash early rather
                        // than silently degrade to plain decode.
                        fail_step(
                            &mut slots,
                            &err.context(format!("GLM5.2 rank {rank} draft")),
                        );
                        break 'serve;
                    }
                };
                if spans.len() != proposal_slots.len() {
                    let err = anyhow::anyhow!(
                        "GLM5.2 rank {rank} draft returned {} spans for {} proposals",
                        spans.len(),
                        proposal_slots.len()
                    );
                    fail_step(&mut slots, &err);
                    break 'serve;
                }
                for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
                    if let Some(active) = slots[rank][slot_id].as_mut() {
                        active.state.set_drafts(span.to_vec());
                    }
                }
            }
        }
    }

    // Also fail whatever never got a slot.
    for req in pending {
        let _ = req.token_tx.send(TokenEvent::Error {
            message: "GLM5.2 engine shut down before the request was scheduled".to_owned(),
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
    }

    // The DeepEP context drop is collective: broadcast Shutdown to every rank
    // BEFORE the workers' Drop joins them one by one — a sequential
    // shutdown-then-join would leave a rank spinning in the destroy barrier
    // for ranks that never got the command (until the ~100 s device timeout).
    for worker in &workers {
        let _ = worker.request_shutdown();
    }
    drop(workers);
}

/// Fast-reject invalid requests at intake (Scheduled → Rejected, the same
/// event order the bs=1 coordinator emitted); valid ones queue for a rank.
fn intake(
    req: GenerateRequest,
    pending: &mut std::collections::VecDeque<GenerateRequest>,
    max_model_len: usize,
    dspark_enabled: bool,
) {
    if let Err(message) = validate_request(&req, max_model_len, dspark_enabled) {
        let prompt_tokens = req.prompt_tokens.len();
        let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
        let _ = req.token_tx.send(TokenEvent::Scheduled {
            queued_at_unix_s,
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens: 0,
        });
        let _ = req.token_tx.send(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens: 0,
        });
        return;
    }
    pending.push_back(req);
}

/// A failed step leaves the ranks permanently out of lockstep: whichever
/// collective the survivors are spinning in would pair with the NEXT step's
/// first dispatch and every layer after it would run against the wrong
/// expert bank — byte-deterministic garbage, no crash. The group cannot be
/// re-synced; fail every active request and tear the engine down.
fn fail_step(slots: &mut [RankSlots], err: &anyhow::Error) {
    log::error!(
        "GLM5.2 step failed; shutting the engine down (the EP8 collective group cannot recover): {err:#}"
    );
    for slot in slots.iter_mut().flatten() {
        let Some(active) = slot.take() else {
            continue;
        };
        let _ = active.req.token_tx.send(TokenEvent::Error {
            message: format!("{err:#}"),
            prompt_tokens: active.req.prompt_tokens.len(),
            completion_tokens: active.state.completion_tokens(),
        });
    }
}

#[cfg(test)]
mod launch_ahead_flag_tests {
    use super::*;

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
        let flags = launch_ahead_flags(&shapes, Some(&shapes), false, true, false, &[], 4096);
        assert!(flags.consume);
    }

    #[test]
    fn slot_handoff_blocks_consume_even_under_identical_shapes() {
        // A finish + admission can reuse a slot id without changing the
        // shape — the desync class the first gate run hit.
        let shapes = vec![shape(1, 1)];
        let flags = launch_ahead_flags(&shapes, Some(&shapes), true, true, false, &[], 4096);
        assert!(!flags.consume);
    }

    #[test]
    fn active_row_count_is_part_of_shape_equality() {
        // Same bucket/slots but a row flipped active <-> pad must not consume:
        // a padding input is not value-distinguishable from an active one.
        let leased = vec![shape(1, 1)];
        let shapes = vec![shape(1, 0)];
        let flags = launch_ahead_flags(&shapes, Some(&leased), false, true, false, &[], 4096);
        assert!(!flags.consume);
    }

    #[test]
    fn no_lease_without_an_empty_queue() {
        let shapes = vec![shape(1, 1)];
        let flags = launch_ahead_flags(&shapes, None, false, false, false, &[], 4096);
        assert!(!flags.lease && !flags.consume);
    }

    /// A standalone `RequestKv` for slot-state tests that never schedule KV
    /// (the pool is leaked so the kvbm internals outlive the test value).
    fn test_kv(prompt: Vec<u32>, max_tokens: usize) -> RequestKv {
        let pool: &'static BlockPool = Box::leak(Box::new(BlockPool::new(PAGE, 64).unwrap()));
        pool.new_request(prompt, max_tokens, None)
    }

    /// One rank holding a single decoding request with the given params (its
    /// prompt token is already fed, so `feed_want() == 1`).
    fn decoding_fleet(params: openinfer_sample::SamplingParams) -> Vec<RankSlots> {
        let (token_tx, _token_rx) = openinfer_core::engine::TokenSink::standalone();
        let req = GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens: vec![10],
            params,
            max_tokens: 8,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        };
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
    fn non_greedy_request_blocks_the_lease() {
        // The speculation feeds each row's argmax token; a sampled row would
        // replay the wrong input, so any non-greedy active blocks the lease.
        let shapes = vec![shape(1, 1)];
        let greedy = decoding_fleet(openinfer_sample::SamplingParams::default());
        assert!(launch_ahead_flags(&shapes, None, false, true, false, &greedy, 4096).lease);

        let sampled = decoding_fleet(openinfer_sample::SamplingParams {
            temperature: 0.7,
            ..Default::default()
        });
        assert!(!launch_ahead_flags(&shapes, None, false, true, false, &sampled, 4096).lease);

        // An effectively-greedy request (top_p nucleus <= 1/vocab holds only
        // the argmax token) takes the argmax path, so it may ride the lease.
        let tiny_top_p = decoding_fleet(openinfer_sample::SamplingParams {
            temperature: 0.7,
            top_p: 0.5 / GLM52_VOCAB as f32,
            ..Default::default()
        });
        assert!(launch_ahead_flags(&shapes, None, false, true, false, &tiny_top_p, 4096).lease);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: &[u32] = &[7];

    fn state(prompt: Vec<u32>, max_tokens: usize, ignore_eos: bool) -> Glm52SlotState {
        Glm52SlotState::new(prompt, max_tokens, ignore_eos, 0)
    }

    /// A standalone `RequestKv` for tests that never schedule KV (the pool
    /// is leaked so the kvbm internals outlive the test value).
    fn test_kv(prompt: Vec<u32>, max_tokens: usize) -> RequestKv {
        let pool: &'static BlockPool = Box::leak(Box::new(BlockPool::new(PAGE, 64).unwrap()));
        pool.new_request(prompt, max_tokens, None)
    }

    fn commit(
        committed: &[u32],
        emit: usize,
        finish: Option<FinishReason>,
        context_rows: usize,
    ) -> Glm52StepOutcome {
        Glm52StepOutcome::Commit {
            committed: committed.to_vec(),
            emit,
            finish,
            context_rows,
        }
    }

    #[test]
    fn prefill_rides_decode_then_emits() {
        let mut state = state(vec![10, 11, 12], 4, false);

        assert_eq!(state.feed_want(), 3);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 10,
                position: 0
            }
        );
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 11,
                position: 1
            }
        );
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);

        // The last prompt token's step yields the first generated token.
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 12,
                position: 2
            }
        );
        assert_eq!(state.advance_span(&[42], EOS), commit(&[42], 1, None, 1));
        assert_eq!(state.completion_tokens(), 1);

        // Decode continues from the emitted token at the next position.
        assert_eq!(state.feed_want(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 3
            }
        );
    }

    #[test]
    fn prompt_span_feeds_consecutive_positions_and_keeps_only_the_last_output() {
        let mut state = state(vec![10, 11, 12, 13], 4, false);

        // One span covers three prompt tokens; mid-prompt outputs discarded.
        assert_eq!(state.feed_want(), 4);
        assert_eq!(
            (0..3).map(|i| state.next_input_at(i)).collect::<Vec<_>>(),
            vec![
                Glm52StepInput {
                    token: 10,
                    position: 0
                },
                Glm52StepInput {
                    token: 11,
                    position: 1
                },
                Glm52StepInput {
                    token: 12,
                    position: 2
                },
            ]
        );
        assert_eq!(
            state.advance_span(&[99, 98, 97], EOS),
            Glm52StepOutcome::Prefilling
        );

        // The next span finishes the prompt; its last output is the first
        // generated token.
        assert_eq!(state.feed_want(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 13,
                position: 3
            }
        );
        assert_eq!(state.advance_span(&[42], EOS), commit(&[42], 1, None, 1));
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn whole_prompt_in_one_span_emits_from_the_boundary_row() {
        let mut state = state(vec![10, 11, 12], 4, false);
        // All three span rows are committed prompt context.
        assert_eq!(
            state.advance_span(&[99, 98, 42], EOS),
            commit(&[42], 1, None, 3)
        );
        assert_eq!(state.completion_tokens(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 3
            }
        );
    }

    #[test]
    fn eos_is_suppressed_and_counts_toward_completion() {
        let mut state = state(vec![10], 4, false);
        assert_eq!(
            state.advance_span(&[7], EOS),
            commit(&[7], 0, Some(FinishReason::Stop), 1)
        );
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn ignore_eos_decodes_through_the_stop_token() {
        let mut state = state(vec![10], 4, true);
        assert_eq!(state.advance_span(&[7], EOS), commit(&[7], 1, None, 1));
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 7,
                position: 1
            }
        );
    }

    #[test]
    fn length_cap_emits_the_final_token() {
        let mut state = state(vec![10], 2, false);
        assert_eq!(state.advance_span(&[42], EOS), commit(&[42], 1, None, 1));
        assert_eq!(
            state.advance_span(&[43], EOS),
            commit(&[43], 1, Some(FinishReason::Length), 1)
        );
        assert_eq!(state.completion_tokens(), 2);
    }

    #[test]
    fn eos_outranks_the_length_cap() {
        let mut state = state(vec![10], 1, false);
        assert_eq!(
            state.advance_span(&[7], EOS),
            commit(&[7], 0, Some(FinishReason::Stop), 1)
        );
    }

    #[test]
    fn max_tokens_one_emits_then_finishes() {
        let mut state = state(vec![10, 11], 1, false);
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.advance_span(&[42], EOS),
            commit(&[42], 1, Some(FinishReason::Length), 1)
        );
    }

    #[test]
    fn verify_span_commits_accepted_prefix_plus_correction() {
        let mut state = state(vec![10], 32, false);
        // Boundary emits the anchor t0 = 20.
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        state.set_drafts(vec![21, 22, 99, 98, 97, 96, 95]);

        // The proposal is truncated to GLM52_DSPARK_SPAN_DRAFTS: a 4-row
        // verify span (anchor + 3 drafts) at consecutive positions.
        assert_eq!(state.feed_want(), 4);
        assert_eq!(
            (0..3).map(|i| state.next_input_at(i)).collect::<Vec<_>>(),
            vec![
                Glm52StepInput {
                    token: 20,
                    position: 1
                },
                Glm52StepInput {
                    token: 21,
                    position: 2
                },
                Glm52StepInput {
                    token: 22,
                    position: 3
                },
            ]
        );

        // Target agrees with drafts 21, 22, diverges at the third (30 != 99):
        // commit the accepted prefix + the correction, context = anchor + 2
        // accepted rows.
        let outputs = [21, 22, 30, 0];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 22, 30], 3, None, 3)
        );
        assert_eq!(state.completion_tokens(), 4);
        // The correction is the next anchor; drafts were consumed.
        assert_eq!(state.decode_anchor(), Some((30, 4)));
        assert_eq!(state.feed_want(), 1);
    }

    #[test]
    fn verify_span_truncated_by_the_planner_accepts_only_fed_drafts() {
        let mut state = state(vec![10], 32, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        state.set_drafts(vec![21, 22, 23, 24, 25, 26, 27]);

        // The planner granted only 3 of the 4 wanted rows: the span is the
        // anchor + first 2 drafts, and acceptance ranges over those 2 only.
        assert_eq!(state.feed_want(), 4);
        let outputs = [21, 22, 23];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 22, 23], 3, None, 3)
        );
        assert_eq!(state.completion_tokens(), 4);
        assert_eq!(state.decode_anchor(), Some((23, 4)));
    }

    #[test]
    fn eos_inside_the_committed_run_truncates_and_finishes() {
        let mut state = state(vec![10], 32, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        // Draft 2 is the EOS token (7): accepted, counted, suppressed; the
        // rest of the committed run is dropped.
        state.set_drafts(vec![21, 7, 23, 24, 25, 26, 27]);
        let outputs = [21, 7, 23, 24];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 7], 1, Some(FinishReason::Stop), 4)
        );
        assert_eq!(state.completion_tokens(), 3);
    }

    #[test]
    fn length_cap_truncates_the_verify_want_and_the_committed_run() {
        let mut state = state(vec![10], 4, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        state.set_drafts(vec![21, 22, 23, 24, 25, 26, 27]);
        // remaining = 3 -> the span may commit at most 3 more tokens.
        assert_eq!(state.feed_want(), 3);
        let outputs = [21, 22, 23];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 22, 23], 3, Some(FinishReason::Length), 3)
        );
        assert_eq!(state.completion_tokens(), 4);
    }

    #[test]
    fn wants_drafts_only_with_two_tokens_of_budget() {
        let mut state = state(vec![10], 3, false);
        assert!(!state.wants_drafts(), "mid-prefill never wants drafts");
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        assert!(state.wants_drafts());
        assert_eq!(state.advance_span(&[21], EOS), commit(&[21], 1, None, 1));
        assert!(!state.wants_drafts(), "one-token tail is a plain row");
    }

    #[test]
    fn sampling_row_is_the_committed_row_of_the_span() {
        let mut state = state(vec![10, 11, 12], 4, false);
        // Mid-prompt span: outputs discarded, nothing to sample.
        assert_eq!(state.sampling_row(2), None);
        // Prompt-completing span: the last row's output is the first
        // generated token.
        assert_eq!(state.sampling_row(3), Some(2));
        assert_eq!(
            state.advance_span(&[99, 98, 42], EOS),
            commit(&[42], 1, None, 3)
        );
        // Plain decode: the single anchor row.
        assert_eq!(state.sampling_row(1), Some(0));
    }

    fn request(
        prompt: Vec<u32>,
        params: openinfer_sample::SamplingParams,
        max_tokens: usize,
    ) -> GenerateRequest {
        let (token_tx, _token_rx) = openinfer_core::engine::TokenSink::standalone();
        GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens: prompt,
            params,
            max_tokens,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        }
    }

    fn sampled(temperature: f32) -> openinfer_sample::SamplingParams {
        openinfer_sample::SamplingParams {
            temperature,
            ..Default::default()
        }
    }

    #[test]
    fn non_greedy_is_rejected_only_with_the_drafter() {
        let req = request(vec![10], sampled(0.7), 4);
        assert!(validate_request(&req, 4096, false).is_ok());
        assert!(validate_request(&req, 4096, true).is_err());
    }

    #[test]
    fn malformed_sampling_params_die_at_intake() {
        // Values the sampler kernel would reject with an `ensure!` — which
        // past intake means a failed step and a whole-engine teardown.
        let cases = [
            openinfer_sample::SamplingParams {
                top_p: 0.0,
                ..sampled(0.8)
            },
            openinfer_sample::SamplingParams {
                top_p: 1.5,
                ..sampled(0.8)
            },
            openinfer_sample::SamplingParams {
                top_p: f32::NAN,
                ..sampled(0.8)
            },
            sampled(f32::INFINITY),
            sampled(f32::NAN),
            openinfer_sample::SamplingParams {
                min_p: 1.0,
                ..sampled(0.8)
            },
            openinfer_sample::SamplingParams {
                min_p: -0.1,
                ..sampled(0.8)
            },
        ];
        for params in cases {
            let req = request(vec![10], params, 4);
            assert!(
                validate_request(&req, 4096, false).is_err(),
                "params must be rejected at intake: {params:?}"
            );
        }
        // The greedy path never reaches the sampler: out-of-range values that
        // ride a greedy request stay accepted (temperature 0 ignores top_p).
        let req = request(
            vec![10],
            openinfer_sample::SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            4,
        );
        assert!(validate_request(&req, 4096, false).is_ok());
    }

    #[test]
    fn collect_sampling_rows_marks_each_spans_committed_row() {
        // Bucket 8: slot 0 decodes (non-greedy), slot 1 finishes its prompt
        // with a 3-row span (non-greedy), slot 3 is mid-prompt (non-greedy,
        // span does NOT complete), slot 2 decodes greedily, slots 4-5 pad.
        let shape = Glm52StepShape {
            bucket: 8,
            slots: [0, 1, 1, 1, 3, 2, 4, 5],
            active_rows: 6,
        };
        let mut rank_slots: RankSlots = std::array::from_fn(|_| None);

        let mut decode_state = state(vec![10], 8, false);
        assert_eq!(
            decode_state.advance_span(&[20], EOS),
            commit(&[20], 1, None, 1)
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
        // Slot 0's decode row is step row 0 (one committed token so far →
        // request-local step 1); slot 1's boundary span commits its LAST row
        // (row 1 + offset 2 = 3, first generated token → step 0). Slot 3's
        // mid-prompt span and slot 2's greedy row contribute nothing.
        assert_eq!(picked, vec![(0, 1), (3, 0)]);
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
    fn decode_anchor_is_the_latest_token_at_its_feed_position() {
        let mut state = state(vec![10, 11], 4, false);
        assert_eq!(state.decode_anchor(), None);
        assert_eq!(
            state.advance_span(&[99, 42], EOS),
            commit(&[42], 1, None, 2)
        );
        // The anchor is what the next decode row would feed.
        assert_eq!(state.decode_anchor(), Some((42, 2)));
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 2
            }
        );
        assert_eq!(state.advance_span(&[43], EOS), commit(&[43], 1, None, 1));
        assert_eq!(state.decode_anchor(), Some((43, 3)));
    }

    fn occ(counts: &[usize]) -> Vec<[bool; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| slot < c))
            .collect()
    }

    /// `counts` decode-phase requests per rank (each wants one row).
    fn decode_wants(counts: &[usize]) -> Vec<[usize; GLM52_MAX_BATCH_PER_RANK]> {
        counts
            .iter()
            .map(|&c| std::array::from_fn(|slot| usize::from(slot < c)))
            .collect()
    }

    /// `admission_target` with an unconstrained pool budget — the pure
    /// occupancy-placement behavior.
    fn target(occupied: &[[bool; GLM52_MAX_BATCH_PER_RANK]]) -> Option<(usize, usize)> {
        let committed = vec![0usize; occupied.len()];
        let usable = vec![usize::MAX; occupied.len()];
        admission_target(occupied, &committed, &usable, 1)
    }

    #[test]
    fn admission_prefers_least_loaded_rank_then_lowest_slot() {
        // Empty fleet: rank 0, slot 0.
        assert_eq!(target(&occ(&[0, 0, 0])), Some((0, 0)));
        // Rank 1 is the least loaded.
        assert_eq!(target(&occ(&[2, 1, 2])), Some((1, 1)));
        // Tie between ranks 0 and 2 → lowest rank id.
        assert_eq!(target(&occ(&[1, 2, 1])), Some((0, 1)));
        // A hole in the middle of a rank's slots is reused first.
        let mut holey = occ(&[3, 3]);
        holey[1][1] = false;
        assert_eq!(target(&holey), Some((1, 1)));
        // Full fleet: no target.
        assert_eq!(target(&occ(&[GLM52_MAX_BATCH_PER_RANK; 2])), None);
    }

    #[test]
    fn admission_respects_the_pool_budget() {
        // Rank 0 has free slots but its pool is fully reserved; rank 1 (more
        // loaded but with budget) takes the request. No rank fits → defer.
        let occupied = occ(&[1, 2]);
        assert_eq!(
            admission_target(&occupied, &[90, 40], &[100, 100], 20),
            Some((1, 2))
        );
        assert_eq!(
            admission_target(&occupied, &[90, 90], &[100, 100], 20),
            None
        );
        // Exact fit admits.
        assert_eq!(
            admission_target(&occupied, &[80, 90], &[100, 100], 20),
            Some((0, 1))
        );
    }

    #[test]
    fn lifetime_blocks_counts_the_dangling_token() {
        // 64 prompt + 1 max_tokens: the generated token is appended to the
        // sequence (dangling) and provisions page 2 even though its KV is
        // never written.
        assert_eq!(lifetime_blocks(64, 1), 2);
        assert_eq!(lifetime_blocks(63, 1), 1);
        assert_eq!(lifetime_blocks(64, 64), 2);
        assert_eq!(lifetime_blocks(64, 65), 3);
    }

    /// Drive one request end to end through the coordinator's exact
    /// schedule/apply sequence against `pool` — the offline replica of the
    /// two engine-fatal submit-walk assertions (span start == `kv_position`,
    /// schedule never fails under the admission reservation). Verify spans
    /// fully accept their drafts, maximizing the KV draw per round. Returns
    /// the first schedule failure (the tight-budget control asserts one).
    fn drive_request(
        pool: &BlockPool,
        prompt_len: usize,
        max_tokens: usize,
        with_drafts: bool,
    ) -> Result<(), String> {
        let prompt: Vec<u32> = (0..prompt_len as u32).map(|t| 10_000 + t).collect();
        let mut state = Glm52SlotState::new(prompt.clone(), max_tokens, true, 0);
        let mut kv = pool.new_request(prompt, max_tokens, None);
        let mut fresh = 60_000u32;
        loop {
            if with_drafts && state.wants_drafts() {
                state.set_drafts(vec![70_001, 70_002, 70_003, 70_004, 70_005, 70_006, 70_007]);
            }
            let span = state.feed_want().min(GLM52_MAX_BATCH_PER_RANK);
            assert_eq!(
                state.next_input_at(0).position,
                kv.kv_position(),
                "span start drifted from the pool's kv_position"
            );
            let mid_prefill = state.fed < state.prompt.len();
            if mid_prefill {
                kv.schedule_prefill(span, pool)
                    .map_err(|e| format!("schedule_prefill: {e}"))?;
            } else if span == 1 {
                kv.schedule_decode(pool)
                    .map_err(|e| format!("schedule_decode: {e}"))?;
            } else {
                kv.schedule_speculative(span, pool)
                    .map_err(|e| format!("schedule_speculative: {e}"))?;
            }
            // The prologue's page-row coverage, offline: the exact page row
            // must cover every fed position.
            let pages = kv.step_page_indices(span);
            let last_position = state.next_input_at(span - 1).position;
            assert!(
                pages.len() * PAGE > last_position,
                "page row misses a fed position"
            );
            fresh += 1;
            // Rows 1.. echo the fed tokens (a verify span fully accepts its
            // drafts), the last row emits a fresh token.
            let outputs: Vec<u32> = (1..span)
                .map(|offset| state.next_input_at(offset).token)
                .chain(std::iter::once(fresh))
                .collect();
            match state.advance_span(&outputs, &[]) {
                Glm52StepOutcome::Prefilling => {
                    kv.apply_prefill_chunk(pool).expect("apply_prefill_chunk");
                }
                Glm52StepOutcome::Commit {
                    committed, finish, ..
                } => {
                    if mid_prefill {
                        kv.apply_prefill(committed[0], pool).expect("apply_prefill");
                    } else if span == 1 {
                        kv.apply_decode(committed[0], pool).expect("apply_decode");
                    } else {
                        kv.apply_speculative(&committed, pool)
                            .expect("apply_speculative");
                    }
                    if finish.is_some() {
                        break;
                    }
                }
            }
        }
        kv.release().map_err(|e| format!("release: {e}"))?;
        Ok(())
    }

    #[test]
    fn full_lifetime_reservation_covers_kvbm_peak_draw() {
        // The submit walk turns any schedule failure into an engine
        // teardown; this is that contract's offline test. A pool sized
        // exactly `lifetime_blocks + 1` (padding) must carry every shape end
        // to end — and one block less must NOT, or the reservation is merely
        // sufficient by accident, not tight.
        for &(prompt_len, max_tokens) in &[
            (64usize, 64usize),
            (64, 65),
            (63, 65),
            (1, 128),
            (127, 2),
            (192, 3),
            (65, 1),
        ] {
            for with_drafts in [false, true] {
                let lifetime = lifetime_blocks(prompt_len, max_tokens);
                let pool = BlockPool::new(PAGE, lifetime + 1).expect("pool");
                drive_request(&pool, prompt_len, max_tokens, with_drafts).unwrap_or_else(|e| {
                    panic!("({prompt_len},{max_tokens},drafts={with_drafts}): {e}")
                });
                let tight = BlockPool::new(PAGE, lifetime).expect("tight pool");
                assert!(
                    drive_request(&tight, prompt_len, max_tokens, with_drafts).is_err(),
                    "({prompt_len},{max_tokens},drafts={with_drafts}): a budget below the \
                     lifetime must fail somewhere"
                );
            }
        }
    }

    #[test]
    fn eos_truncated_speculative_apply_stays_in_contract() {
        // EOS mid-verify-span truncates `committed` (the suppressed EOS is
        // its last entry); `apply_speculative` with the truncated run and
        // the release must both stay clean.
        let pool = BlockPool::new(PAGE, 16).expect("pool");
        let prompt: Vec<u32> = (0..70).collect();
        let mut state = Glm52SlotState::new(prompt.clone(), 32, false, 0);
        let mut kv = pool.new_request(prompt, 32, None);
        loop {
            if state.fed >= state.prompt.len() {
                break;
            }
            let span = state.feed_want().min(GLM52_MAX_BATCH_PER_RANK);
            assert_eq!(state.next_input_at(0).position, kv.kv_position());
            kv.schedule_prefill(span, &pool).expect("schedule_prefill");
            match state.advance_span(&vec![50u32; span], EOS) {
                Glm52StepOutcome::Prefilling => {
                    kv.apply_prefill_chunk(&pool).expect("apply_prefill_chunk");
                }
                Glm52StepOutcome::Commit { committed, .. } => {
                    kv.apply_prefill(committed[0], &pool)
                        .expect("apply_prefill");
                }
            }
        }
        state.set_drafts(vec![21, 7, 23]);
        let span = state.feed_want();
        assert_eq!(span, 4, "anchor + 3 drafts");
        assert_eq!(state.next_input_at(0).position, kv.kv_position());
        kv.schedule_speculative(span, &pool)
            .expect("schedule_speculative");
        let outcome = state.advance_span(&[21, 7, 23, 99], EOS);
        let Glm52StepOutcome::Commit {
            committed,
            emit,
            finish,
            ..
        } = outcome
        else {
            panic!("verify span must commit");
        };
        assert_eq!(committed, vec![21, 7], "truncated to the consumed run");
        assert_eq!(emit, 1, "the suppressed EOS is consumed, not emitted");
        assert_eq!(finish, Some(FinishReason::Stop));
        kv.apply_speculative(&committed, &pool)
            .expect("apply_speculative with the truncated run");
        kv.release().expect("release");
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

    #[test]
    fn cached_prefix_starts_feeding_at_the_suffix() {
        // 3 blocks of prompt with the first 2 cache-hit: feeding starts at
        // position 128 and only the suffix is ever fed.
        let prompt: Vec<u32> = (0..192).collect();
        let s = Glm52SlotState::new(prompt, 8, false, 128);
        assert_eq!(s.feed_want(), 64);
        assert_eq!(s.next_input_at(0).position, 128);
        assert_eq!(s.next_input_at(0).token, 128);
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
    fn bucket_is_the_smallest_covering_the_hungriest_rank() {
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[0, 0]))),
            vec![(1, vec![0]), (1, vec![0])]
        );
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[1; 8]))),
            vec![(1, vec![0]); 8]
        );
        // One rank at two requests lifts EVERY rank to the 2-row bucket —
        // idle ranks pad with free slots.
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[2, 1]))),
            vec![(2, vec![0, 1]), (2, vec![0, 1])]
        );
        assert_eq!(
            forwarded(&plan_step_shapes(&decode_wants(&[3, 1])))[0],
            (4, vec![0, 1, 2, 3])
        );
        // Past the 4-row bucket the full batch takes over.
        assert_eq!(forwarded(&plan_step_shapes(&decode_wants(&[5, 1])))[0].0, 8);
    }

    #[test]
    fn partial_buckets_pack_actives_first() {
        // A rank holding slots {1, 5} forwards them in rows 0..2; the padding
        // rows (bucket 4) ride on the lowest free slots.
        let mut holey = decode_wants(&[0, 3]);
        holey[0][1] = 1;
        holey[0][5] = 1;
        assert_eq!(
            forwarded(&plan_step_shapes(&holey)),
            vec![(4, vec![1, 5, 0, 2]), (4, vec![0, 1, 2, 3])]
        );
        let mut deep = decode_wants(&[5, 0]);
        deep[0][0] = 0;
        deep[0][7] = 1;
        assert_eq!(
            forwarded(&plan_step_shapes(&deep))[0],
            (8, vec![1, 2, 3, 4, 7, 0, 5, 6])
        );
    }

    #[test]
    fn prefill_want_extends_one_slot_into_a_span() {
        // A lone mid-prefill request with plenty of prompt left fills the
        // whole max bucket with its span; idle ranks pad.
        let mut wants = decode_wants(&[0, 0]);
        wants[0][2] = 3000;
        let shapes = plan_step_shapes(&wants);
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
            forwarded(&plan_step_shapes(&wants))[0],
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
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![0, 1, 1, 1, 1, 1, 1, 1])
        );

        // Two mid-prefill slots with small wants: both met, remaining rows
        // pad on free slots.
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 3;
        wants[0][1] = 2;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
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
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![2, 2, 2, 2, 5, 5, 5, 5])
        );

        // A decode slot in the mix keeps its single row; the prefills split
        // what remains (7 rows -> 4 + 3 by round-robin order).
        let mut wants = decode_wants(&[0]);
        wants[0][0] = 1;
        wants[0][3] = 3000;
        wants[0][6] = 3000;
        assert_eq!(
            forwarded(&plan_step_shapes(&wants))[0],
            (8, vec![0, 3, 3, 3, 3, 6, 6, 6])
        );
    }
}
