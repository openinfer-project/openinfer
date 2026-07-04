# Cargo feature-flag convention

**TL;DR**: One feature name per model line, spelled identically in `openinfer-server` and `openinfer-kernels`. A model crate itself carries **no** self-named feature — unless its kernels need a build-time Python toolchain, in which case the *only* allowed gate is a single whole-crate `#![cfg(feature = "...")]`. Capability features (`moe`, `pplx-ep`, `kernel-report`, …) are named after the capability, never after a model.

## Why features exist here at all

A feature answers exactly one question: **does this binary compile this model line's kernels?** Everything else — which model actually runs, EP topology, backend selection — is runtime configuration, not a feature. Two forces shape the rules:

1. `openinfer-kernels/build.rs` is expensive (nvcc, and for some lines AOT codegen through Python). Features scope what it compiles.
2. Two model lines cannot build without external toolchains (`qwen35-4b`: Python + Triton; `deepseek-v4`: Python + TileLang + CuTe DSL). A featureless `cargo build --workspace` must stay buildable on a machine that has CUDA but no Python — so those two, and only those two, stay gated at the crate level.

## The rules

### 1. Feature names

The feature for a model line is the crate-name suffix: `openinfer-kimi-k2` → `kimi-k2`. The same string is used in `openinfer-server` and `openinfer-kernels`. Capability features describe the capability (`moe`, `pplx-ep`, `kernel-call-trace`, `kernel-report`, `deepseek-v4-cutedsl-diagnostic`), and may imply model features but are never a synonym for one.

### 2. `openinfer-kernels` — one feature per model's kernels

Each model line gets a same-named feature that scopes its `csrc/<model>/` sources (and AOT codegen, if any). Shared substrate gets a capability feature that model features imply (`glm52 = ["moe"]`, `kimi-k2 = ["moe"]`). Features that need more than nvcc must say so in a comment next to the declaration.

### 3. Model crates — Tier A (pure CUDA): no self feature

Applies to: **qwen3, glm52, kimi-k2, deepseek-v2-lite** — anything whose kernels need only nvcc.

- No self-named feature. The crate always compiles its full self.
- `openinfer-kernels = { workspace = true, features = ["<model>"] }`, **not** optional.
- No `#[cfg(feature = "<model>")]` anywhere in the crate, no reject-only stub functions, no `required-features` on tests/bins (except for genuine capability features like `kernel-report`).
- Cost accepted knowingly: workspace-wide builds (`cargo test --workspace --lib`) compile these kernels once, then cache. That trade was made in #550 (glm52) and extended to kimi-k2 / deepseek-v2-lite.
- Corollary: without `required-features`, a bare `cargo test --workspace` (no `--lib`) also builds and *runs* Tier A GPU gates, which fail loudly on hosts missing the weights or GPU count — same as the qwen3/glm52 gates today. The routine sweep is `--lib`; gates run per package.

### 4. Model crates — Tier B (build-time Python toolchain): one whole-crate gate

Applies to: **qwen35-4b** (Triton), **deepseek-v4** (TileLang + CuTe DSL).

- Keep the self-named feature, forwarding to the kernels feature: `deepseek-v4 = ["openinfer-kernels/deepseek-v4"]`.
- The *only* in-crate gate is `#![cfg(feature = "<model>")]` at the top of `lib.rs`. With the feature off, the crate compiles to nothing. Scattered item-level `#[cfg]` gates and reject-only stubs are forbidden — they rot into the load-only scaffold #550 had to remove.
- Tests/bins/benches declare `required-features = ["<model>"]`.

### 5. `openinfer-server` — the selection point

- Tier A: `<model> = ["dep:openinfer-<model>"]`.
- Tier B: `<model> = ["dep:openinfer-<model>", "openinfer-<model>/<model>"]`.
- `default = ["qwen3"]` keeps the stock build pure Rust + CUDA.
- Server code touches a model crate only under `#[cfg(feature = "<model>")]`; the "rebuild with --features" error message lives in the server dispatch (`server_engine.rs`), not in model crates.

## What this means in practice

| Command | Features in play |
| --- | --- |
| `cargo run --release -- --model-path …` | server `qwen3` (default) |
| `cargo run --release --features kimi-k2 -- …` | server feature; the kimi crate needs no flag of its own |
| `cargo test --release -p openinfer-kimi-k2` | none needed — Tier A crates build whole |
| `cargo test --release -p openinfer-qwen35-4b --features qwen35-4b --test hf_golden_gate` | Tier B: the self feature is required |
| `cargo test --release --workspace --lib` | compiles Tier A kernels (incl. `moe` substrate → needs NCCL), skips Tier B (no Python required) |

Feature unification note: cargo compiles `openinfer-kernels` once per distinct feature set in a build graph. Default-member builds (server + qwen3) and `--workspace` builds are two different sets; both stay cached side by side in `target/`, so switching between them does not thrash nvcc after the first build of each.

## Adding a new model line

1. Add the kernels feature (`<model> = []` or `= ["moe"]`) and gate its `csrc/<model>/` sources in `build.rs`.
2. Decide the tier by one question: *does building the kernels need anything beyond nvcc?* No → Tier A. Yes → Tier B, and document the toolchain requirement next to the feature declaration.
3. Wire the server feature per rule 5 and add the `#[cfg]` arms in `server_engine.rs` / `main.rs`.
4. Bring-up scaffolding (load-only binaries, reject-only coordinators) is fine on a branch but must not merge behind a crate-internal feature — see #550 for the cleanup that pattern forces later.
