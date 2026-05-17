This file provides guidance to Coding Agent when working with code in this repository.

## What is pegainfer

Pure Rust + CUDA LLM inference engine (~7K Rust, ~3.4K CUDA). No PyTorch, no frameworks. Supports Qwen3-4B, Qwen3-8B, and Qwen3.5-4B (hybrid linear + full attention). OpenAI-compatible `/v1/completions` API.

## Build & Run

**Always use `--release`** — debug builds are extremely slow for GPU/CUDA and will timeout.

```bash
cargo run --release -- --model-path models/Qwen3.5-4B
```

**Key env vars:**
- `PEGAINFER_CUDA_SM` — GPU SM target override when `nvidia-smi` unavailable (e.g. `120` or `120,80`)
- `PEGAINFER_TRITON_PYTHON` — Python with Triton for build-time AOT kernel generation
- `PEGAINFER_TEST_MODEL_PATH` — override test model path (default: `models/Qwen3-4B`)

## Tests

```bash
# Unit tests (~9s)
cargo test --release --workspace --lib

# E2E greedy regression — requires GPU + model weights
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release -p pegainfer-qwen3-4b --test e2e
PEGAINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p pegainfer-qwen35-4b --test e2e

# Single test
cargo test --release embedding_variants -- --nocapture
```

E2E tests compare against JSON baselines in `test_data/`. Regenerate baselines after any change that affects numerical output.

## Architecture

```
HTTP Request → vLLM frontend → EngineHandle → per-model scheduler/executor → TokenEvent
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
              pegainfer-qwen3-4b                         pegainfer-qwen35-4b
               (full attention)                       (24 linear + 8 full attn)
                         │                                           │
                         └─────────────────────┬─────────────────────┘
                                               │
                         pegainfer-core runtime + pegainfer-kernels
                                               │
                                 CUDA / cuBLAS / Triton / FlashInfer
```

**Key abstractions:**

- **`pegainfer-core::engine`** — shared request/event contract (`EngineHandle`, `GenerateRequest`, `TokenEvent`) used by the server and model crates.
- **Per-model crates** — Qwen3 and Qwen3.5 own config, weights, prefill/decode/unified execution, scheduler, tests, and benches.
- **`pegainfer-core::ops`** — shared GPU operator wrappers used by model crates.
- **`pegainfer-kernels`** — tensor/FFI/kernel build owner for CUDA, cuBLAS wrappers, FlashInfer wrappers, and Triton AOT outputs.
- **CUDA Graph** — decode path captured inside model executors with pre-allocated buffers to preserve pointer stability.
- **KV state** — model schedulers own request state; shared paged-KV primitives live in `pegainfer-core`.

**Build system**: the virtual workspace root has no package build script. `pegainfer-kernels/build.rs` owns CUDA/Triton compilation:
1. Compiles `pegainfer-kernels/csrc/*.cu` with nvcc (auto-detects GPU SM targets)
2. Runs Triton AOT via `pegainfer-kernels/tools/triton/gen_triton_aot.py` for Qwen3.5 compatibility kernels

---

# Team Documentation Workflow

Collaboration centered on the `docs/` directory.

## Knowledge Architecture (domain-axis)

Docs are organized by what they're *about*, not by lifecycle stage. A doc's freshness lives in its TL;DR (and `Last touched:` for active areas) — not by which directory it sits in. Completed work stays co-located with its domain. There is no `archives/` directory — if a doc no longer earns its keep, delete it; if a lasting lesson hides inside it, lift that lesson into `lessons/` first, then delete.

```
docs/
├── index.md           # Routing table — every doc must be listed here
├── roadmap/           # Strategic plans, quarterly direction, milestones
├── models/<line>/     # Per-model living docs (qwen3, qwen35, deepseek-v4, ...)
│                      # — design, accuracy, perf, refactor records, gotchas
├── subsystems/<area>/ # Cross-cutting components (runtime, scheduler, frontend, kernels)
├── playbooks/         # Reusable how-to: benching, profiling, accuracy, onboarding
├── lessons/           # Tribal knowledge from research / other projects
├── benchmarks/        # Standalone benchmark snapshots and eval reports
├── conventions/       # Ongoing standards (bench regression, coding style)
└── private/           # Local-only notes (gitignored)
```

Classification rule at capture time:
- Is it tied to a specific model? → `models/<line>/`
- A specific subsystem? → `subsystems/<area>/`
- Reusable how-to applicable across models? → `playbooks/`
- Lasting lesson from elsewhere (other repo, research, postmortem)? → `lessons/`
- Snapshot of measurement, not a doc that evolves? → `benchmarks/`
- Strategic / cross-cutting plan? → `roadmap/`

If you can't pick one, the doc probably needs splitting.

## Documentation Style

- Docs cover what `--help` and code can't: pitfalls, diagnostic paths, decision context. Don't restate CLI reference.
- Every command in a doc must be run and verified before committing. Unverified commands are technical debt.
- The only required header is a one-line **TL;DR**. Keep it true; that's the contract.
- For `models/<line>/` and `subsystems/<area>/` docs, add `Last touched: YYYY-MM` and bump it when you do meaningful work on the doc (not for typo fixes). The date is a fact, not a judgement — readers infer freshness themselves.
- `playbooks/`, `lessons/`, `conventions/`, `roadmap/`, `benchmarks/`, `archives/` don't need a freshness stamp. They're either timeless until disproven, or self-dated, or explicitly inert.
- No `Status:` enum. Enum fields go stale exactly when you need them most.

## index.md Drift Policy

`index.md` is a routing table with a scanning-friendly TL;DR column. It is *allowed to drift* from the TL;DR inside each doc — the doc body is authoritative. Update `index.md` when you create or delete a doc, or when the existing TL;DR is so wrong it actively misleads. Don't churn it on every doc edit.

## Core Principles (CODE)

Documentation exists to advance work, not to hoard information. Four steps when handling information:

1. **Capture**: Only record what materially advances the project. When in doubt, leave it out.
2. **Organize**: Action-oriented. Resist the urge to organize for organization's sake — structure should be just enough.
3. **Distill**: Refactor over append. When you learn something new or hit a pitfall, integrate it into the document body — don't pile a changelog at the bottom.
4. **Express**: Every document must point to a next step. Split unwieldy documents proactively. Active documents must note the current blocker or next action.

## Collaboration Lifecycle

**Sync**

At the start of each session, you must read `index.md` and load the documents needed for the task at hand.

**Execute**
- Update relevant documents as you go. When a new problem or idea arises, create a document in the appropriate domain directory (see classification rule above).
- Record *why* a decision was made, not just *what* was done.

**Commit**

When a session wraps up:
- Update the TL;DR (and `Last touched`, where applicable) at the top of each modified document.
- Update `index.md` only when you created or deleted a doc, or when its TL;DR row is now misleading (see Drift Policy above).

---

# Git Conventions

Commit messages use Commitizen format: `<type>(<scope>): <subject>`. Never commit directly to `main` — create a `feat/`/`fix/`/`chore/`/… branch first.

# Code Conventions

Module files use the flat layout (`src/ops.rs` + `src/ops/`) — no `mod.rs`.
