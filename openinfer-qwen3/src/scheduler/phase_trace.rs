//! Per-request host-side phase tracing for the scheduler.
//!
//! Emits `queue` → `prefill` → `decode` spans as children of the request span
//! the frontend opened (passed via [`GenerateRequest::trace_parent`]). Unlike
//! attributing phases from event arrival at the frontend demux — where the
//! `Scheduled` event and first token land microseconds apart and prefill looks
//! like ~0 — these spans are opened and closed *where the work happens* on the
//! scheduler thread, so their durations are the real host-side phase times.
//!
//! Live [`Span`]s are non-`Clone`, so they never enter the scheduler's `Clone`
//! request state; they live here in a side-table keyed by [`RequestId`]. The
//! request's [`SpanContext`] (which *is* `Copy`) rides through request state and
//! is handed to the tracker at admission.
//!
//! Every method is a cheap no-op when tracing is off: the parent context is
//! `None`, so no span is ever created and the table stays empty.

use std::collections::HashMap;

use fastrace::Span;
use fastrace::collector::SpanContext;

use crate::executor::RequestId;

/// The phase a request's open span currently represents. Guards against
/// double-transitions (e.g. a chunked prefill spanning several steps must open
/// `prefill` once, not once per chunk).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queue,
    Prefill,
    Decode,
}

struct Tracked {
    parent: SpanContext,
    phase: Phase,
    /// The currently-open phase span. Dropping it ends the phase; reassigning
    /// closes the old phase and opens the new one.
    span: Span,
}

/// Side-table of in-flight request phase spans. Owned by the scheduler loop.
#[derive(Default)]
pub(super) struct PhaseTracker {
    tracked: HashMap<RequestId, Tracked>,
}

impl PhaseTracker {
    /// Begin tracing a request in the `queue` phase. `parent` is the frontend's
    /// request span context; `None` (tracing off) makes this and every later
    /// call a no-op for this request.
    pub(super) fn enter_queue(&mut self, id: RequestId, parent: Option<SpanContext>) {
        let Some(parent) = parent else { return };
        let span = Span::root("queue", parent);
        self.tracked.insert(
            id,
            Tracked {
                parent,
                phase: Phase::Queue,
                span,
            },
        );
    }

    /// The request's prompt work first reached the GPU: close `queue`, open
    /// `prefill`. Idempotent across chunked-prefill steps.
    pub(super) fn enter_prefill(&mut self, id: RequestId) {
        if let Some(t) = self.tracked.get_mut(&id) {
            if t.phase != Phase::Queue {
                return;
            }
            t.phase = Phase::Prefill;
            t.span = Span::root("prefill", t.parent);
        }
    }

    /// The first token was produced: close `prefill`, open `decode`.
    pub(super) fn enter_decode(&mut self, id: RequestId) {
        if let Some(t) = self.tracked.get_mut(&id) {
            if t.phase == Phase::Decode {
                return;
            }
            t.phase = Phase::Decode;
            t.span = Span::root("decode", t.parent);
        }
    }

    /// The request finished (or was dropped/rejected): close its open span and
    /// stop tracking it. Dropping the whole request trace is the frontend root
    /// span's job; this only ends the last phase span.
    pub(super) fn finish(&mut self, id: RequestId) {
        self.tracked.remove(&id);
    }
}
