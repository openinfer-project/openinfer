# GLM5.2 cross-node DP16: replicated deterministic coordinator (SMR)

> **TL;DR:** Design sketch, no code yet. To scale the DP8 lock-step scheduler across machines (DP16 = 2×8 H200), do NOT remote the coordinator into a hub-and-spoke RPC master. Instead exploit the fact that `scheduler.rs` is already a deterministic state machine over two input streams (admission order + per-step outputs): **replicate the coordinator on every node**, have a sequencer assign a total order to requests, allgather each step's output tokens (≤8 u32/rank, piggybacking on a GPU path that already runs 16 MoE all-to-alls), and let every replica independently derive byte-identical step shapes with zero per-step cross-node control traffic. Placement (`admission_target`) moves out of the router into the replicated state machine; the router only sequences. Prerequisite discipline: every non-deterministic input the coordinator currently reads locally (`token_tx.is_closed()` disconnect probes) must become an explicit event in the sequenced stream. The fallback plan — framed-TCP hub-and-spoke — is fine at today's 20–46 ms steps (<1% RTT overhead) and is the pragmatic first cut; SMR is the endgame if steps get fast, node count grows, or we want the deterministic replay log it produces for free.
>
> **Last touched:** 2026-07

## Where this comes from

The DP8 coordinator (`openinfer-glm52/src/scheduler.rs`, see `dp8-scheduler.md` and
`continuous-batching.md`) is a single in-process loop: it owns all 8 ranks' slot state,
plans one global step shape per step (`plan_step_shapes`), fans the step out to 8
`Glm52RankWorker` threads over crossbeam channels, joins all 8 responses, and only then
decides the next step. There is deliberately **no per-rank scheduler layer** — the DeepEP
contract (every rank enters every MoE dispatch/combine with the same global row count)
makes every scheduling decision global, so per-rank discretion has nothing to decide.

That works because everything is one process on one node. DP16 across two nodes breaks
exactly one assumption: the coordinator↔worker channel is no longer free.

## The naive translation and its three walls

The direct port is hub-and-spoke: keep one coordinator on node A, replace crossbeam
channels to node B's workers with a network transport.

1. **Transport.** Control messages are tiny (a step command is ~8 rows of
   `(token, position)` + shape + flags, a few hundred bytes; a response is ≤8 u32) and
   slow-cadence (one per 20–46 ms step). Latency budget is generous — sub-ms RTT is <1%
   of a step. The semantic requirements are the interesting part: reliable, **strict
   per-connection FIFO** (the coordinator relies on channel FIFO order today, e.g. draft
   commands ordering before the next step), and **fail-stop** (a lost message = collective
   desync = byte-deterministic garbage; the engine must tear down, not retry).
2. **EP collectives.** DP16 means EP16: every MoE layer's dispatch/combine crosses the
   IB fabric instead of NVLink. DeepEP supports internode (NVSHMEM/IBGDA), so this is
   feasible, but it is the real performance wall — and it belongs to `openinfer-comm`,
   not the scheduler. This doc does not solve it; it only requires that whatever the
   comm layer does, all ranks still agree on the bucket every step.
3. **Fault model.** Any step failure already tears the whole engine down (the EP group
   cannot re-sync). Cross-node, network flaps and single-node maintenance join the list
   of fleet-fatal events. Nothing in this doc improves availability; wide-EP deployments
   elsewhere (vLLM/SGLang) share the property.

### Transport pick for the hub-and-spoke fallback

Evaluated against the "tiny, slow, FIFO, fail-stop" profile:

- **RDMA (verbs)** — buys µs latency and zero-copy bandwidth; we need neither on the
  control plane. Costs QP state machines, memory registration, and error semantics that
  are strictly worse than a TCP reset for fail-stop detection. Rebuilding "reliable
  ordered pipe with clean failure" on verbs is reinventing TCP with worse observability.
  RDMA stays where it belongs: the DeepEP data plane.
- **gRPC** — pays for schema evolution, multi-client service discovery, LB, TLS; this is
  a fixed-membership internal pipe between one binary and itself. The tonic/tower/h2
  stack is dead weight and hides our frames inside HTTP/2 framing during debugging.
- **ZMQ** — its headline features (auto-reconnect, silent buffering, HWM drops) are the
  exact inverse of fail-stop. vLLM/SGLang use it as Python path-of-least-resistance, not
  because the semantics fit. We would spend effort disabling its resilience.
- **Framed TCP + serde (bincode/postcard), one persistent connection per remote worker**
  — the kernel gives reliable + strict FIFO + connection-reset-as-explicit-failure for
  free; `connection reset` maps 1:1 onto `fail_step`. The code shape is a mechanical
  swap: `crossbeam::Sender<Command>` → `write(len ‖ bincode(cmd))`; the per-call response
  channel becomes in-order response matching (the protocol is already exactly-one-response
  -per-command, in order). `TCP_NODELAY` on. Two mandatory patches: a version handshake
  at connect (git hash — internal protocols don't do schema evolution, so reject mismatch
  at the door), and per-step response timeouts at seconds granularity (the step cadence
  *is* the heartbeat; TCP keepalive's minutes-scale default is useless, and a 5 s timeout
  reports root cause two orders of magnitude before DeepEP's ~100 s device trap).

This fallback is acceptable indefinitely at current step times. The rest of the doc is
the design that makes the per-step control RTT disappear entirely.

## The SMR design

### The observation

`scheduler.rs` is already factored as a deterministic state machine:

- `Glm52SlotState` — per-request pure data transitions;
- `admission_target`, `plan_step_shapes`, `launch_ahead_flags` — pure functions over
  `[rank][slot]` snapshots;
- the coordinator loop — a thin impure shell moving tokens between channels.

The entire scheduling state is a deterministic function of exactly two input streams:

1. **the admission stream** — which requests, in what order, effective at which step
   boundary;
2. **the step-output stream** — every rank's output tokens each step (they decide who
   finishes, which slots free, what the next feed-wants are).

Same code + same two streams ⇒ every replica computes byte-identical `slots`, shapes,
and launch-ahead flags forever. This is textbook deterministic state machine
replication — the Calvin/deterministic-DB shape: **sequence first, then execute with
zero coordination**.

### The architecture

```
                 ┌────────────────────────────┐
   requests ───► │ sequencer (HTTP frontend)  │   assigns a total order (LSN);
                 │  "R joins at step N+2"     │   does NOT pick rank/slot
                 └──────┬──────────────┬──────┘
              ordered   │              │   ordered
              admission │              │   admission
              stream    ▼              ▼   stream
        ┌──────────────────┐   ┌──────────────────┐
        │ node A            │   │ node B            │
        │ coordinator       │   │ coordinator       │
        │ replica           │   │ replica           │
        │  · full 16-rank   │   │  · full 16-rank   │
        │    slot mirror    │   │    slot mirror    │
        │  · drives ranks   │   │  · drives ranks   │
        │    0..8 locally   │   │    8..16 locally  │
        │    (crossbeam,    │   │    (crossbeam,    │
        │     as today)     │   │     as today)     │
        └───────┬──────────┘   └───────┬──────────┘
                │      per-step token allgather      │
                └──────────── (≤8 u32 × 16 ranks, ───┘
                     piggybacks the GPU collective path)
```

- **The sequencer is the router, demoted.** It keeps exactly one intelligence: stamping
  a total order on incoming requests (and cancel events — see below). It does *not*
  choose a rank or slot. Placement is `admission_target`, computed identically inside
  every replica. Broadcast alone is not enough — it must be **ordered** broadcast
  (atomic broadcast); a reordering between replicas forks `admission_target`'s input and
  desyncs the fleet.
- **Admission takes effect at a declared step boundary.** Replicas must agree on *which*
  step a request joins, so the sequencer stamps "R joins at step N+2" — one step of
  slack to propagate the record off the critical path, the same trick launch-ahead uses.
- **Every replica simulates the whole fleet, executes only its slice.** `plan_step_shapes`
  needs the global feed-want picture (the bucket is set by the hungriest rank, which may
  live on the other node), so each replica maintains all 16 ranks' `slots`. The
  difference is execution: node A submits `step_async` only to its local 8 workers; the
  other 8 ranks' state advances by replaying the allgathered output tokens.
- **Step outputs are allgathered, not routed through a master.** ≤8 u32 per rank per
  step. The GPU critical path already runs 16 MoE all-to-alls per step; one more tiny
  allgather (or piggybacking the last combine) is noise. After it, both replicas hold
  identical step-output vectors and fold them into identical state.
- **Tokens stream to clients from the owner only.** A request's client connection
  terminates on one node; only that replica writes real `TokenEvent`s. The other replica
  advances the same `Glm52SlotState` as shadow bookkeeping (slot occupancy, feed-wants)
  with its emission sinked. In DB terms: every replica replays the full log; each serves
  reads only for its own shard.

### The determinism discipline

A replica may not read one bit of node-local state on the decision path. The current
code has exactly one known violation: **`token_tx.is_closed()`** — the client-disconnect
probe (used both for the prefill zombie-slot fix and the decode send-failure path). The
client hangs off one node; the other replica cannot observe the disconnect and would
keep the slot occupied ⇒ divergent shapes ⇒ collective desync.

Fix: disconnects become **cancel events in the sequenced stream**. Only the owner
replica observes the dead sink; it injects "cancel R effective at step N+2" through the
sequencer like any admission record, and both replicas free the slot at the same
boundary. The same rule generalizes: wall-clock, queue-emptiness-at-an-instant,
randomness — all either leave the decision path or become sequenced events.
(`queued_at_unix_s` timestamps are emission-only metadata today, not decision inputs —
they stay local.) The pure-function factoring means this discipline is cheap to keep:
the impure surface is confined to the coordinator shell, and the shell is exactly what
gets rewritten.

### What SMR buys and what it doesn't

- **Buys:** zero per-step cross-node control RTT — each node's control decisions are
  local, and cross-node sync happens only where it is physically unavoidable (the
  collectives). Coordination cost drops from per-step to per-request.
- **Buys, for free:** a **deterministic replay log**. Persist the two streams (admission
  records + per-step output vectors) and any desync, scheduling bug, or
  "byte-deterministic garbage, no crash" incident can be replayed through the
  coordinator state machine on a laptop — no 16 GPUs required. For the failure class
  where the fleet produces wrong output with no error, this is essentially the only
  post-hoc forensic tool.
- **Does not buy:** availability. A failed step still kills the fleet; replicas exist
  for locality, not redundancy. The sequencer is also a single point of failure exactly
  as the HTTP frontend already is.

## Decision: when to build which

| condition | build |
| --- | --- |
| DP16 bring-up, steps 20–46 ms | framed-TCP hub-and-spoke — RTT is <1% of a step; smallest diff, scheduler untouched |
| steps optimized to single-digit ms, or >2 nodes (RTT tail grows with fan-out) | SMR replicas |
| we want deterministic replay for desync forensics regardless of node count | the sequenced-stream part of SMR stands alone: even the single-node DP8 coordinator can log (admission order, step outputs) today and gain replay debugging |

The third row is the cheapest first step and is valuable independent of DP16: a
`(admission, outputs)` journal behind a flag on the current coordinator, plus a replay
harness that drives `scheduler.rs` state transitions from the journal and asserts shape
determinism. It de-risks the determinism discipline (any hidden nondeterminism shows up
as replay divergence on one node, not collective garbage on sixteen) before any network
code exists.

## Next step

None scheduled — DP16 is not on the current roadmap; this doc exists so the design
discussion isn't lost. If cross-node work starts, begin with the replay journal
(third row above), then the framed-TCP fallback, and treat SMR as the step-time-driven
upgrade.
