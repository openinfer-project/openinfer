# GLM5.2 native MTP accuracy and acceptance

> **TL;DR:** The c8 native-MTP acceptance gap was a correctness bug at the target/draft boundary:
> OpenInfer passed the target's pre-final-norm residual to MTP, while official vLLM passes the
> model-returned final-normalized hidden. Using `scratch.final_normed` raises the matched c8 mean
> accepted length from `1.753` to `3.725` versus official vLLM's `3.786`, and reduces TPOT from
> `23.44 ms` to `11.31 ms`. On the selected 251-token target trajectory, mean accepted length
> changes from `1.000` to `5.795`. The remaining c8 acceptance delta is `0.061` (`1.6%`).
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
  5. Re-run online accepted-length and c8 TPOT A/B on the same 8×H200 node. Treat an optimization as
     a win only when measured accepted length improves without target-quality regression.
- **Risks / open questions**:
  - Official vLLM runs TP8+EP8 while the current OpenInfer path is TP1/DP8+EP8. Target raw hidden can
    differ from collective and accumulation order even when final task quality is healthy.
  - The existing five-row oracle proves MTP front/layer-78 behavior on official vLLM inputs, but
    does not cover a long online teacher-forced trajectory or the production paged-KV recycle path.
  - Near-tie top logits must not be reported as a structural model bug without regret/margin data.

### C8 accepted-length decomposition follow-up

- **Question**: Which component accounts for the measured `3.786` versus `1.753` c8 accepted-length
  gap: different target trajectories, MTP forward numerics, or state carried between draft steps?
- **Plan**:
  1. Reconstruct the seeded 64-request token-ID corpus used by the retained benchmark and persist
     it with tokenizer, seed, and checksum metadata. Feed identical token IDs to both engines so
     prompt generation is no longer a hidden variable.
  2. Run the engines sequentially on the same 8×H200 node and retain per-request target tokens,
     draft tokens, accepted-prefix lengths, and rejection margins. Stratify requests by low,
     median, and high acceptance before collecting tensor checkpoints.
  3. For each selected request, stop at the first differing target token or first differing draft
     token. Compare the raw target hidden supplied to MTP before comparing MTP prepare output,
     first-step logits, recycled normalized hidden, and layer-78 KV for draft steps 2–5.
  4. Replay official target hidden through the OpenInfer MTP oracle at those proposal-entry states.
     This intervention separates target-model drift from the MTP forward without requiring the two
     target implementations to remain on the same greedy path.
  5. Classify the aggregate gap by counterfactual: measure how many acceptance decisions recover
     under a shared target trajectory and official-hidden replay. Only then change code, add the
     narrowest production-path regression gate, and rerun the matched c8 benchmark.
- **Interpretation gates**:
  - Target raw hidden differs first: report target topology/numerics and quantify acceptance under
    official-hidden replay; do not attribute the gap to MTP forward.
  - Target raw hidden agrees but the first draft logits differ materially: localize MTP prepare,
    layer 78, final norm, and shared head in that order.
  - First draft agrees but a later draft diverges: inspect recycled hidden and MTP paged-KV
    positions at the first differing step.
  - Use top-1 regret, target/draft margins, RMS, and p99. Exact equality is not required when the
    selected token and acceptance decision are stable.
- **Operational risks**:
  - Official vLLM and OpenInfer consume the same eight GPUs, so they must run sequentially. Compact
    artifacts must be saved before switching engines, and the clean OpenInfer service must be
    restored afterward.
  - Dumping full hidden states for `64 × 256` tokens is excessive. Only first-difference states from
    the stratified subset should be retained; aggregate runs keep tokens, counters, and margins.
  - TP8+EP8 versus TP1/DP8+EP8 may prevent bit parity even on a teacher-forced path. Conclusions
    must distinguish harmless numeric drift from changes to selected tokens or accepted prefixes.

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
- The retained c8 artifacts for the same 64 random requests, plus the corrected replay, report:

  | Engine | Mean TPOT | Mean accepted length |
  | --- | ---: | ---: |
  | official vLLM native MTP | `16.90 ms` | `3.786` |
  | OpenInfer before hidden-boundary fix | `23.44 ms` | `1.753` |
  | OpenInfer after hidden-boundary fix | `11.31 ms` | `3.725` |

  The OpenInfer values are weighted from the final 64 per-request histograms: `9191` rounds before
  the fix and `4346` rounds after it.
- Result: the corrected A/B confirms that shorter accepted prefixes caused most of the observed
  TPOT gap. `TPOT × accepted length` is `41.10 ms` before and `42.11 ms` after the fix, consistent
  with unchanged speculative-round cost. The post-fix OpenInfer/vLLM TPOT comparison still
  includes engine and topology differences; it is not a claim that the MTP round itself is faster.

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
- Corrected OpenInfer replay: benchmark JSON SHA-256
  `5e4c3d7219fee985008e6932bdb07c74708b5a39691480259db357e6ece2db55`;
  histogram record SHA-256
  `6c95eb444cbf8296773be94b425d2b83899b47e718f2c42e13dab0a5995fb2ba`.
  The aggregate histogram is `[795, 650, 469, 852, 201, 1379, 0, 0]`: `4346` rounds and `11843`
  accepted drafts. Mean accepted length is `1 + 11843 / 4346 = 3.725035`.

### C8 first-difference diagnosis and fix

- Reconstructed the retained 64-request corpus from the benchmark version that produced the
  original artifact. Its token-ID corpus SHA-256 is
  `59516edc33d2a7a36b63628db7cd4eb0888f3aec7f9ac97b49039920743928d3`; it contains `8152`
  prompt tokens and requests exactly `16384` output tokens.
- Per-round target traces disprove the earlier working hypothesis that target-trajectory drift
  explains the full acceptance gap. Six requests share at least 32 target tokens across engines.
  An exploratory capture found one 251-token shared trajectory: before the fix OpenInfer rejects
  every first draft (`251` rounds, mean accepted length `1.000`), while official vLLM reached
  approximately `5.78`. That full official per-request capture was not retained as a durable
  artifact, so the approximate value is diagnostic rather than a regression baseline.
- At the first proposal of that request, the embedding is bit-exact but the target hidden passed
  into MTP differs before any layer-78 or MTP-KV work:

  | Boundary | Before-fix cosine | After-fix cosine |
  | --- | ---: | ---: |
  | target hidden supplied to MTP | `0.8361` | `0.9818` |
  | `eh_proj` output | `0.8320` | `0.9771` |
  | layer-78 raw hidden | `0.7404` | `0.9877` |
  | recycled normalized hidden | `0.7171` | `0.9861` |

- The before-fix target-hidden norm is `281.01`, versus official vLLM's `78.63`. OpenInfer's
  post-fix norm is `77.32`. This large discontinuity is not attributable to BF16 reduction order.
- Official vLLM registers `GlmMoeDsaForCausalLM` on its DeepSeek-V2-compatible path. That target
  returns final-RMSNorm hidden states, and the MTP proposer consumes that model return directly.
  OpenInfer instead selected `scratch.hidden`, the residual before final RMSNorm. The fix changes
  the source to `scratch.final_normed`; token shifting, layer 78, MTP KV, and the verifier remain
  unchanged.
- On the selected shared trajectory, the first draft changes from the incorrect token `98863` to
  official vLLM's `98825`. The corrected OpenInfer run records 44 rounds with accepted-draft
  histogram `[1, 1, 0, 0, 0, 42, 0, 0]`, or mean accepted length `5.795`.
- The retained short official trace has SHA-256
  `02e97a2f4d23b573cf53b20611840dc098e5f694bfde9284f532bda2c972d999`. Its six captured
  proposal records reject the first two drafts, then fully accept all five drafts in the next four
  records. The ordered checksum-list SHA-256 for the retained official tensor trace is
  `ee07cf72de367ffcb03d543bf8be4ea7ff9bb9bb7fa1d42a08cf1b25ba52461c`.
  The OpenInfer tensor dumps used for the cosine table were transient, so those tensor metrics are
  diagnostic rather than a standalone reproducible gate.
- The corrected selected-request response and acceptance log have SHA-256
  `8f5a6663d6ddb10c59458a1f6849fff468150265443398da63aa492d0b3ca3f3` and
  `5807097280c2edf83a60b4f1841760ca60e7061dddad6916edb49a5835093d94`.
- Release validation passes `83` library tests with `19` GPU/model gates ignored, followed by the
  two explicitly enabled official-vLLM MTP front and EP8 layer-78 golden gates. A new ignored
  production-path gate loads the full checkpoint on 8×H200, submits the selected 256-token request
  through the real scheduler/executor, and asserts target trajectory, first draft `98825`, and mean
  accepted length at least `5.0`; it passes in `210.37 s`. The clean c8 replay completes `64/64`
  requests with the exact retained input/output totals.

### Pre-fix fixed-prompt comparison

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
- A suspected MTP-step raw-hidden recycle bug was rejected. Official vLLM traces show that each
  next draft step receives the prior `shared_head`-normalized hidden bit-exactly; OpenInfer already
  feeds the same normalized value.
- Result: this narrow prompt excluded a token shift, stale recycle hidden, and an obvious MTP
  KV-position error, but did not validate the target-to-MTP input boundary. Its first proposal
  happened to agree despite the wrong source tensor. The broader c8 first-difference trace above
  supersedes the earlier topology-numerics attribution.

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

### How to use aggregate accepted length

- The matched random benchmark uses identical input/output lengths, but official vLLM runs a
  TP8+EP8 target while OpenInfer runs TP1/DP8+EP8. Their greedy generated texts diverge from the
  first request, so each MTP instance is scored on a different output trajectory.
- Accepted length is strongly content-dependent. Comparing `3.40` to `2.00` therefore describes
  the measured serving systems and suggests one contributor to their TPOT difference, but does
  not isolate MTP forward correctness or quantify compute-side differences.
- Aggregate acceptance alone still cannot identify the faulty component. Here, per-request target
  traces found long shared trajectories and tensor comparison found the first difference before
  MTP forward. After that attribution, the matched c8 acceptance result is a useful regression
  measurement.

## Debrief

- **Outcome**: Native MTP now consumes the same target-hidden boundary as official vLLM. Matched c8
  acceptance is `3.725` versus `3.786`, and the formerly pathological shared trajectory aligns.
  Task-level quality remains healthy on the measured GSM8K slice.
- **Pitfalls encountered**:
  - A small offline layer-78 match cannot certify online accepted length because it bypasses the
    production target hidden-state source and long-lived MTP KV.
  - Calling both tensors “target hidden” concealed a material API boundary: the residual before
    final RMSNorm and the model-returned hidden are not interchangeable MTP inputs.
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
  - Keep the reconstructed c8 corpus as a serving-level regression measurement. Use the selected
    production-path gate for the target/MTP handoff so a future change from `final_normed` back to
    the pre-norm residual fails before benchmark interpretation.
