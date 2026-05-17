# DeepSeek V4 Kernel Paths

**Created**: 2026-05-09
**Status**: complete - CUDA path, TileLang path, and DeepSeek V4 kernel routing index are organized and verified on default plus DeepSeek feature paths.

## Preparation

- **Read**:
  - `docs/index.md` - showed DeepSeek V4 support, kernel boundary, and Qwen3 kernel extraction as the relevant prior work.
  - `docs/models/deepseek-v4/support.md` - confirmed DeepSeek V4 currently has native MP8 runtime, TileLang build-time kernels, exact E2E coverage, and a documented CUDA split by subsystem.
  - `docs/subsystems/kernels/pegainfer-kernels-boundary.md` - confirmed kernels belong in the shared kernels crate, while model DAG/runtime policy stays in the model crate.
  - `docs/models/qwen3/kernels-crate.md` - established the existing crate-first split and the role of `pegainfer-kernels/KERNELS.md`.
  - `docs/conventions/coding-style.md` - reminded that GPU kernels deserve targeted tests, while broad behavior is better covered by integration/E2E.
  - `pegainfer-kernels/build.rs` - showed DeepSeek kernels are feature-gated by filename prefix in a flat `csrc/` scan, and TileLang generation was hard-coded to the old flat `tools/tilelang/gen_deepseek_v4_tilelang.py` path.
  - `pegainfer-kernels/KERNELS.md` - currently indexes Qwen3 and only mentions DeepSeek as compatibility symbols, so DSV4 has no routing table.
  - `pegainfer-kernels/csrc/deepseek_*.cu` and `pegainfer-kernels/csrc/deepseek_common.cuh` - confirmed the CUDA side is already split by subsystem but still lives in the root kernel source directory.
  - `pegainfer-deepseek-v4/src/runtime/*` - confirmed runtime calls reach DeepSeek symbols through `pegainfer_kernels::ffi`, so path cleanup should not require runtime API changes.
- **Relevant history**:
  - `docs/models/deepseek-v4/support.md` records that the current DeepSeek CUDA glue is intentionally split by subsystem; this cleanup should preserve that split instead of merging files.
  - `docs/models/qwen3/kernels-crate.md` moved kernel ownership into `pegainfer-kernels`; the same pattern supports moving model-specific source into a clearer subdirectory without changing model runtime ownership.
- **Plan**:
  1. First slice: move DeepSeek V4 CUDA sources from `pegainfer-kernels/csrc/deepseek_*.cu` and `deepseek_common.cuh` into `pegainfer-kernels/csrc/deepseek_v4/`, then update `pegainfer-kernels/build.rs` to discover CUDA files recursively and feature-gate DeepSeek by path instead of flat filename prefix.
  2. Keep object file names stable or explicitly namespace them so `ar` input names remain collision-free when sources live in subdirectories.
  3. Update include/rerun handling so `.cu` and `.cuh` changes under nested kernel directories trigger rebuilds.
  4. Run low-cost verification for the first slice: `cargo fmt --all --check`, `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-kernels --features deepseek-v4`, and the non-DeepSeek default check if local CUDA/TileLang availability permits.
  5. Record the result in this doc, then decide the next slice: likely moving the TileLang generator into a DeepSeek-specific tools path and adding a DSV4 section to `pegainfer-kernels/KERNELS.md`.
- **Risks / open questions**:
  - Recursive source discovery can accidentally compile generated or third-party CUDA if scoped too broadly. It should only recurse under owned `csrc/`.
  - DeepSeek TileLang requires a working TileLang Python; local verification may stop at environment setup rather than code correctness.
  - Default builds must continue skipping DeepSeek CUDA when the `deepseek-v4` feature is disabled.

## Execution Log

### Step 1: Move DeepSeek V4 CUDA sources under a model-specific directory
- Moved DeepSeek V4 CUDA sources from `pegainfer-kernels/csrc/deepseek_*.cu` and `pegainfer-kernels/csrc/deepseek_common.cuh` into `pegainfer-kernels/csrc/deepseek_v4/`.
- Updated `pegainfer-kernels/build.rs` to collect owned `csrc/` files recursively, emit rebuild triggers for nested `.cu`/`.cuh` files, and generate object names from the relative source path so nested CUDA files do not collide with flat ones.
- Replaced the build-script feature probe with `cfg!(feature = "deepseek-v4")`. Cargo feature resolution was checked with `cargo tree -p pegainfer-server --features deepseek-v4 -i pegainfer-kernels -e features`, which confirmed `pegainfer-server/deepseek-v4` enables `pegainfer-deepseek-v4/deepseek-v4` and then `pegainfer-kernels/deepseek-v4`.
- Updated `pegainfer-kernels/KERNELS.md` and `docs/models/deepseek-v4/support.md` to point DeepSeek CUDA references at `csrc/deepseek_v4/`.

Result: path move and build-script feature gating are in place.

Verification:
- `cargo fmt --all --check` passed.
- `cargo tree -p pegainfer-server --features deepseek-v4 -i pegainfer-kernels -e features` confirmed feature forwarding from server to DeepSeek V4 model crate to kernels.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-kernels` passed. The build log confirmed DeepSeek V4 CUDA/TileLang kernels are disabled without the feature.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-kernels --features deepseek-v4` passed. The build log confirmed DeepSeek V4 TileLang CUDA generation under `target/.../tilelang/deepseek_v4/`.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-server --features deepseek-v4` passed, covering feature forwarding through the server, model crate, and kernels crate together.

### Step 2: Move the DeepSeek V4 TileLang generator under the shared TileLang backend directory
- Moved `pegainfer-kernels/tools/tilelang/gen_deepseek_v4_tilelang.py` to `pegainfer-kernels/tools/tilelang/deepseek_v4/generate.py`.
- Updated `pegainfer-kernels/build.rs` to run the generator from the new path.
- Updated the generated CUDA banner comment to point at the new generator path.
- Added `pegainfer-kernels/tools/tilelang/README.md` to define `tools/tilelang/` as the shared TileLang backend directory, with model- or shape-family-specific generators in subdirectories.
- Updated `docs/models/deepseek-v4/support.md` to point at the new generator path.

Verification:
- `cargo fmt --all --check` passed.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-kernels --features deepseek-v4` passed. The build log showed DeepSeek V4 TileLang CUDA generation still succeeds after the path move.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-server --features deepseek-v4` passed after the generator move.

### Step 3: Add the DeepSeek V4 kernel routing index
- Added a `DeepSeek V4 MP8 Path` section to `pegainfer-kernels/KERNELS.md`.
- Mapped runtime owners under `pegainfer-deepseek-v4/src/runtime/` to the public `pegainfer_kernels::ffi` symbols and their CUDA/TileLang source owners.
- Grouped rows by execution subsystem rather than every individual shape: quant, attention, collectives cast helpers, indexer, compressor, HC, logits, and MoE.
- Kept TileLang shape details in source notes so the table remains a routing aid rather than a duplicate ABI declaration.

Verification:
- `rg` over `pegainfer-kernels/src/ffi.rs`, `pegainfer-kernels/csrc/deepseek_v4/`, and `pegainfer-deepseek-v4/src/runtime/` was used to build the mapping.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-kernels` passed without `deepseek-v4`; the build log confirmed DeepSeek V4 CUDA/TileLang kernels are disabled on the default path.
- `PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-server` passed without `deepseek-v4`; the build log again confirmed the default server path skips DeepSeek V4 CUDA/TileLang.

## Debrief

- **Outcome**: DeepSeek V4 owned CUDA sources now live under `pegainfer-kernels/csrc/deepseek_v4/`. The DeepSeek V4 TileLang generator now lives under the shared TileLang backend directory at `pegainfer-kernels/tools/tilelang/deepseek_v4/generate.py`. The kernels build script recursively scans owned CUDA sources, skips DSV4 by path when `deepseek-v4` is disabled, uses `cfg!(feature = "deepseek-v4")` for the feature decision, namespaces object names by relative source path, and runs the TileLang generator from its new path. `pegainfer-kernels/KERNELS.md` now includes a DeepSeek V4 MP8 routing table from runtime owners to FFI symbols and CUDA/TileLang source paths.
- **Pitfalls encountered**:
  - The initial feature probe used `CARGO_FEATURE_DEEPSEEK_V4`; Cargo already forwards the feature into `pegainfer-kernels`, so `cfg!(feature = "deepseek-v4")` is clearer in `build.rs`.
  - `cargo tree -e features` needs the reverse dependency form to show the exact feature forwarding path clearly.
- **Lessons learned**:
  - Moving model-owned kernel source into a subdirectory is low-risk once build discovery is path-based rather than filename-prefix based.
  - TileLang is better represented as a shared backend directory with model-specific generator subdirectories, not as a DeepSeek-owned top-level tools directory.
- **Follow-ups**:
  - Consider adding Rust-side DeepSeek kernel wrapper modules only after there is a repeated call pattern worth centralizing; direct FFI remains acceptable for the current model-owned runtime.
