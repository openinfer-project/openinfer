This file provides guidance to Coding Agent when working with code in this repository.

## What is openinfer

Pure Rust + CUDA LLM inference engine. No PyTorch, no frameworks. OpenAI-compatible `/v1/completions` API.

**Supported models:**

Every model line is behind a cargo feature; only `qwen3` is a default feature, so the stock build is pure Rust + CUDA with no Python.

| Model | Crate | Feature flag | Architecture |
|-------|-------|-------------|-------------|
| Qwen3-4B / 8B | `openinfer-qwen3` | `qwen3` (default) | Full attention, TP support |
| Qwen3.5-4B | `openinfer-qwen35-4b` | `--features qwen35-4b` (needs build-time Python + Triton) | 24 linear + 8 full attention |
| DeepSeek-V2-Lite | `openinfer-deepseek-v2-lite` | `--features deepseek-v2-lite` | MoE + EP, 2-GPU |
| Kimi-K2 | `openinfer-kimi-k2` | `--features kimi-k2` | MLA + MoE + Marlin INT4, 8-GPU EP |
| GLM5.2 | `openinfer-glm52` | `--features glm52` | MLA + MoE + FP8, 8-GPU EP (bring-up) |

## Build & Run

**Always use `--release`** — debug builds are extremely slow for GPU/CUDA and will timeout.

```bash
# Qwen3 (default feature, no Python anywhere in the build)
cargo run --release -- --model-path models/Qwen3-4B

# Feature-gated models
cargo run --release --features qwen35-4b -- --model-path models/Qwen3.5-4B
cargo run --release --features kimi-k2 -- --model-path models/Kimi-K2
cargo run --release --features deepseek-v2-lite -- --model-path models/DeepSeek-V2-Lite
cargo run --release --features glm52 -- --model-path models/GLM5.2
```

**Key env vars:**
- `OPENINFER_CUDA_SM` — GPU SM target override when `nvidia-smi` unavailable (e.g. `120` or `120,80`)
- `OPENINFER_TRITON_PYTHON` — Python with Triton for `qwen35-4b` build-time AOT kernel generation (falls back to `.venv/bin/python`, then `python3`, then `python`)
- `OPENINFER_TILELANG_PYTHON` — Python with TileLang for the `glm52` sparse-MLA build-time AOT (sm_90a targets only)
- `OPENINFER_NCCL_ROOT` — NCCL root (>= 2.30.4) for DeepEP shim (`moe` feature)
- `OPENINFER_FLASHINFER_INCLUDE` — FlashInfer include dir override
- `OPENINFER_TEST_MODEL_PATH` — override test model path (default: `models/Qwen3-4B`)
- `OPENINFER_BUILD_TIMING=1` — print per-phase build timings (nvcc, Triton AOT, etc.)
- `OPENINFER_NVCC_JOBS` — override parallel nvcc job count

## Tests

```bash
# Unit tests (~9s)
cargo test --release --workspace --lib

# Accuracy and integration tests — require GPU + model weights
cargo test --release -p openinfer-qwen3 --test hf_golden_gate
OPENINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test hf_golden_gate
OPENINFER_TEST_MODEL_PATH=models/Qwen3.5-4B cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test e2e_scheduler

# Single test (filter by name)
cargo test --release --workspace --lib prefix_cache -- --nocapture
```

Qwen accuracy gates compare logits against stored HF golden fixtures. Qwen3.5 exact-text JSON baselines are retired; keep `e2e_scheduler` for scheduler liveness and request-flow coverage.

## Architecture

```
HTTP Request → vLLM frontend → EngineHandle → per-model scheduler/executor → TokenEvent
                                               │
              ┌──────────┬─────────────┬───────┼───────────┬──────────┐
              │          │             │       │           │          │
        openinfer-  openinfer-   openinfer-  openinfer-  openinfer-  ...
        qwen3       qwen35-4b    dsv2-lite   kimi-k2     glm52
      (full attn) (linear+full) (MoE+EP)   (MLA+MoE)  (MLA+MoE+FP8)
              │          │             │       │           │          │
              └──────────┴─────────────┴───────┼───────────┴──────────┘
                                               │
                          openinfer-core runtime + openinfer-kernels
                                               │
                               ┌───────────────┼───────────────┐
                               │               │               │
                       CUDA / cuBLAS    Triton AOT      FlashInfer
                                                    (sampling, attention,
                                                     norm, MLA decode)
```

**Key abstractions:**

- **`openinfer-engine`** — shared request/event contract (`EngineHandle`, `GenerateRequest`, `TokenEvent`) used by the server and model crates. (`openinfer-core::engine` re-exports it.)
- **Per-model crates** — each model owns config, weights, prefill/decode execution, scheduler, tests, and benches.
- **`openinfer-core::ops`** — shared GPU operator wrappers used by model crates.
- **`openinfer-kernels`** — tensor/FFI/kernel build owner for CUDA, cuBLAS, FlashInfer, and Triton AOT. Model-specific kernels live in feature-gated submodules (`kimi_k2`, `glm52`).
- **CUDA Graph** — decode path captured inside model executors with pre-allocated buffers to preserve pointer stability.
- **KV state** — model schedulers own request state; shared paged-KV primitives live in `openinfer-kv-cache`; host/SSD/RDMA offload bridge in `openinfer-kv-offload`.

**Build system**: the virtual workspace root has no package build script. `openinfer-kernels/build.rs` owns CUDA/Triton compilation:
1. Compiles `openinfer-kernels/csrc/*.cu` with nvcc (auto-detects GPU SM targets)
2. Feature-gated codegen: `qwen35-4b` runs Triton AOT via `openinfer-kernels/tools/triton/gen_triton_aot.py`; `kimi-k2` adds MLA/MoE/Marlin CUDA; `glm52` adds MLA/MoE/FP8 CUDA plus TileLang sparse-MLA codegen on sm_90a

---

# Team Documentation Workflow

Collaboration centered on the `docs/` directory.

## Knowledge Architecture (domain-axis)

Docs are organized by what they're *about*, not by lifecycle stage. A doc's freshness lives in its TL;DR (and `Last touched:` for active areas) — not by which directory it sits in. Completed work stays co-located with its domain. There is no `archives/` directory — if a doc no longer earns its keep, delete it; if a lasting lesson hides inside it, lift that lesson into `lessons/` first, then delete.

```
docs/
├── index.md           # Routing table — every doc must be listed here
├── roadmap/           # Strategic plans, quarterly direction, milestones
├── models/<line>/     # Per-model living docs (qwen3, qwen35, kimi-k2, ...)
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
