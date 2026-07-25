# GLM5.2 native MTP accuracy and acceptance

> **TL;DR:** Native MTP5 passes a production-path acceptance check against official vLLM: the
> first five-token proposal is exact, and over four comparable rounds the accepted-draft counts
> are `[1,1,1,1]` versus `[1,1,1,2]`. The one acceptance-changing top-1 flip is not accompanied by
> evidence of a token shift or recycle/KV error; its cause is not isolated, and the target token
> stream remains identical.
> Task quality is also stable (`98.0%` versus `97.5%` on the measured GSM8K slice). The lower
> aggregate accepted length is real for the measured OpenInfer serving topology, but it is not a
> standalone MTP correctness failure because the two benchmarks follow different target-model
> greedy trajectories.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — routes GLM5.2 model work and accuracy methodology.
  - `docs/models/glm52/dspark-mtp.md` — establishes accepted-length accounting, span verification,
    and the rule that speculative performance must be explained as round cost divided by accepted
    tokens.
  - `docs/models/glm52/moe-tp8-low-latency.md` — records the existing TP8/EP8 topology and its
    numerical and performance characteristics.
  - `docs/models/glm52/oracle-harness.md` — requires an external truth implementation, seeded
    reproducibility, and RMS/p99 rather than max-only float checks.
  - `docs/playbooks/accuracy-parity-playbook.md` — prescribes exact token IDs, first-diff
    localization, and production-path teacher forcing.
  - `docs/playbooks/bench-vs-vllm.md` — defines matched hardware, model, client, sampling, and
    prefix-cache controls for comparative serving measurements.
- **Relevant history**:
  - DSpark reached useful accepted lengths only after validating the real online hidden-state and
    verify path; a small standalone forward match was not considered sufficient.
  - GLM5.2 bucket/topology changes can move near-tie decisions without breaking task accuracy, so
    token mismatches and structural drift must be classified separately.
- **Plan**:
  1. Choose one deterministic prompt and capture the official vLLM greedy token IDs plus per-step
     target raw hidden, final target logits, and native-MTP intermediate tensors.
  2. Teacher-force those exact token IDs through the OpenInfer production path and capture the same
     checkpoints, without comparing states after the token streams diverge.
  3. Find the first divergent checkpoint in this order: target raw hidden, MTP prepare/first logits,
     then MTP recycle hidden and layer-78 KV across draft steps 2–5.
  4. Classify the first difference as target topology/numerics, MTP forward semantics, or MTP
     paged-KV/recycle state; add the narrowest reproducible accuracy gate before changing code.
  5. Re-run online accepted-length and c1 TPOT A/B on the same 8×H200 node. Treat an optimization as
     a win only when measured accepted length improves without target-quality regression.
- **Risks / open questions**:
  - Official vLLM runs TP8+EP8 while the current OpenInfer path is TP1/DP8+EP8. Target raw hidden can
    differ from collective and accumulation order even when final task quality is healthy.
  - The existing five-row oracle proves MTP front/layer-78 behavior on official vLLM inputs, but
    does not cover a long online teacher-forced trajectory or the production paged-KV recycle path.
  - Near-tie top logits must not be reported as a structural model bug without regret/margin data.

## Execution Log

### Native MTP forward and serving integration

- Added the native layer-78 MTP head, its separate one-layer KV state, proposal/recycle loop, and
  scheduler-wide collective modes for reset-only, context-only, and proposal rounds.
- Verified release build, library and CLI tests, an 8-rank end-to-end run, and an official-vLLM
  fixture gate.
- Result: functional. The official-vLLM fixture comparison measured MLP RMS `8.62e-3`
  (`p99 2.54e-2`), chained hidden RMS `1.87e-2` (`p99 5.86e-2`), exact top-1, top-8 overlap `8/8`,
  and top-32 overlap `30/32`.

### End-task accuracy

- Ran the same GSM8K five-shot evaluation with native MTP disabled and enabled.
- Result: target quality is stable on the measured slice: plain `195/200` (`97.5%`), native MTP
  `196/200` (`98.0%`). This rules out a broad target-quality collapse, not an MTP acceptance issue.

### Serving performance and acceptance attribution

- Matched official vLLM and OpenInfer on one 8×H200 node with greedy decoding and fixed output
  length. An exploratory c1 terminal snapshot measured:

  | Engine | c1 TPOT | Mean accepted length | TPOT × accepted length |
  | --- | ---: | ---: | ---: |
  | official vLLM native MTP | `9.10 ms` | `3.40` | `30.9 ms` |
  | OpenInfer native MTP | `15.03 ms` | `2.00` | `30.1 ms` |

- OpenInfer plain c1 TPOT was `18.66 ms`; native MTP therefore helps, but less than vLLM.
- The OpenInfer accepted-length value is reproducible from eight retained request histograms
  (`1009` speculative rounds), not an average of per-request averages. The standalone c1 client
  result was not retained, however, so the table is mechanism evidence rather than a regression
  baseline.
- The retained c8 artifacts for the same 64 random requests report:

  | Engine | Mean TPOT | Mean accepted length |
  | --- | ---: | ---: |
  | official vLLM native MTP | `16.90 ms` | `3.786` |
  | OpenInfer native MTP | `23.44 ms` | `1.753` |

  OpenInfer's value is weighted from the final 64 per-request histograms (`9191` rounds).
- Result: the exploratory c1 product suggests that shorter accepted prefixes are a major part of
  its TPOT gap, but the unmatched artifact does not exclude speculative-round compute differences.
  The retained c8 comparison confirms the system-level acceptance gap, but cannot isolate MTP
  correctness because the target trajectories differ.

#### Retained c8 measurement record

- Hardware/model: one 8×H200 node, the same GLM5.2 FP8 checkpoint, prefix cache disabled.
- Client workload: random dataset, default seed `0`, nominal input length `128`, output length
  `256`, `64` prompts, concurrency `8`, unlimited request rate, temperature `0`, ignore EOS.
  Both runs completed `64/64`, with `8152` input and `16384` output tokens.
- Official vLLM provenance: commit `dcfebf93`, benchmark JSON SHA-256
  `af909e6a8a06562ff42bfa882decf1462ee0c63771be08b2229201f9b766e780`.
  Raw counters are `4347` proposal rounds, `21735` drafted tokens, and `12109` accepted draft
  tokens. Thus mean accepted length including the target bonus is
  `1 + 12109 / 4347 = 3.785599`.
- OpenInfer provenance: commit `fd6bd6e0`, benchmark JSON SHA-256
  `66fd2d4ea38fa7ceb1429612e35f2f9fce6eed26f35546cfb2da1c0563619760`.
  The aggregate accepted-draft histogram for indices `0..7` is
  `[4564, 3113, 955, 415, 65, 79, 0, 0]`: `9191` rounds and `6923` accepted drafts. Thus mean
  accepted length including the target bonus is `1 + 6923 / 9191 = 1.753237`.

### Production-path acceptance comparison

- Used the deterministic prompt `The capital of France is` with greedy decoding. The official
  trace was captured from vLLM commit `dcfebf93`; OpenInfer ran its normal target, scheduler,
  long-lived MTP KV, and five-step proposal loop.
- The first proposal is exact:

  | Engine | Draft tokens |
  | --- | --- |
  | official vLLM | `[13, 576, 3283, 315, 12089]` |
  | OpenInfer | `[13, 576, 3283, 315, 12089]` |

- Repeating the OpenInfer request four times produced the same proposal and output, ruling out
  request-to-request nondeterminism.
- The first four rounds that share the same target token trajectory accept:

  | Engine | Accepted drafts per round | Mean including target bonus |
  | --- | --- | ---: |
  | official vLLM | `[1, 1, 1, 2]` | `2.25` |
  | OpenInfer | `[1, 1, 1, 1]` | `2.00` |

- The durable proposal record before their round cadence diverges is:

  | Round | official vLLM | OpenInfer |
  | ---: | --- | --- |
  | 0 | `[13, 576, 3283, 315, 12089]` | `[13, 576, 3283, 315, 12089]` |
  | 1 | `[504, 279, 6722, 315, 9621]` | `[504, 279, 3283, 315, 12089]` |
  | 2 | `[311, 7148, 374, 12089, 11]` | `[311, 14915, 409, 93729, 273]` |
  | 3 | `[374, 220, 16, 13, 20]` | `[374, 264, 3283, 315, 220]` |

- The first two proposals agree through the acceptance-relevant prefix. Their first later top-1
  difference is `6722` versus `3283`; official logits are `18.25` versus `18.00`, and that
  position is already behind a rejected draft, so it cannot change accepted length.
- The only acceptance-changing difference in the four comparable rounds is official token `220`
  versus OpenInfer token `264`. In the official BF16 logits, `220` is `18.00` and `264` is
  `17.25`. The target token stream remains identical through the compared interval; only the
  drafter loses one accepted token.
- A suspected raw-hidden recycle bug was rejected. Official vLLM traces show that each next draft
  step receives the prior `shared_head`-normalized hidden bit-exactly; OpenInfer already feeds the
  same normalized value. Feeding raw pre-norm hidden would be incorrect.
- Result: the compared production trajectory shows no evidence of a token shift, stale recycle
  hidden, or MTP KV-position error. The remaining delta is consistent with topology-sensitive
  numerics, but this trace neither attributes that flip nor explains the aggregate gap.

#### Retained fixed-prompt record

- Official side: vLLM commit `dcfebf93`, TP8+EP8, `max_tokens=8`; its five rank-0 proposal groups
  contain 25 paired forward/logit records. The ordered checksum-list SHA-256 is
  `cae3e354d39b21f16d207899d8addc3e12f3b14d1cdb4d4cfd893a0533f4778f`.
- OpenInfer side: commit `fd6bd6e0`, TP1/DP8+EP8, `max_tokens=20` so the short-tail policy exposes
  at least four proposal rounds. The request uses temperature `0` and seed `1`.
- The prompt token IDs are `[6722, 315, 9621, 374, 12089]`. The shared target token trajectory
  through the compared interval is
  `[12089, 13, 31008, 504, 12089, 311, 54831, 374, 220, 101294]`.
- Official vLLM reports four verified rounds, five accepted drafts, and per-position acceptance
  rates `[1.0, 0.25, 0.0, 0.0, 0.0]`; these counters and the proposal transitions reproduce
  accepted drafts `[1,1,1,2]`. OpenInfer's four rounds are reproduced directly from the table and
  the shared target trajectory as `[1,1,1,1]`.

### Why aggregate accepted length is not a cross-engine accuracy gate

- The matched random benchmark uses identical input/output lengths, but official vLLM runs a
  TP8+EP8 target while OpenInfer runs TP1/DP8+EP8. Their greedy generated texts diverge from the
  first request, so each MTP instance is scored on a different output trajectory.
- Accepted length is strongly content-dependent. Comparing `3.40` to `2.00` therefore describes
  the measured serving systems and suggests one contributor to their TPOT difference, but does
  not isolate MTP forward correctness or quantify compute-side differences.
- A hard cross-engine acceptance threshold requires teacher-forcing one retained target token
  trajectory through both production paths. The fixed-prompt trace above is the current retained
  narrow gate; a broad teacher-forced corpus remains a separate harness project, not a prerequisite
  for accepting this implementation.

## Debrief

- **Outcome**: Native MTP is integrated, task-level quality is healthy on the measured GSM8K slice,
  and its production proposal/accept state is reasonably aligned with official vLLM. The c1 data
  suggest lower accepted length contributes to the TPOT gap; the unmatched artifact does not
  establish its share or exclude speculative-round compute differences.
- **Pitfalls encountered**:
  - A small offline layer-78 match cannot certify online accepted length because it bypasses the
    production target hidden-state source and long-lived MTP KV.
  - TPOT alone hid the mechanism; multiplying TPOT by accepted length motivated the acceptance
    investigation, but is not a substitute for a retained per-round timing measurement.
  - Reading the vLLM model return type in isolation suggested that raw hidden might be recycled.
    The captured call boundary proved the opposite: vLLM returns normalized recycle hidden to the
    proposer while retaining raw hidden only for logits.
  - A top-1 mismatch after an earlier rejected draft is diagnostically useful but irrelevant to
    accepted length. First-diff analysis must stop at the effective accepted prefix.
- **Lessons learned**:
  - Native-MTP acceptance is itself an accuracy metric even when target-generated answers remain
    correct.
  - Cross-engine aggregate acceptance is meaningful only when both engines follow the same target
    token trajectory; identical prompt lengths and sampling flags are insufficient.
  - For distributed BF16/FP8 paths, classify top-1 differences with logit margin and whether the
    position can affect acceptance before treating them as structural defects.
- **Follow-ups**:
  - Retain the official-vLLM fixtures and fixed-prompt procedure as the accuracy gate for changes
    to MTP prepare, recycle, sparse-index reuse, and KV positioning.
  - Build a broader teacher-forced production corpus only when acceptance work needs to distinguish
    target-topology drift from drafter regressions statistically.
