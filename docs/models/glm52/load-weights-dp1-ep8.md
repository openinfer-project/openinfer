# GLM5.2 DP1 EP8 Weight Loading

> **TL;DR:** This PR slice adds a load-weight-only GLM5.2 path from latest main: rank0 loads non-expert tensors plus experts 0..31, ranks1..7 load 32 routed experts each, optimized real-checkpoint load measured `63420ms` first run / `50803ms` immediate repeat with rank-local slabs, coalesced H2D ranges, and an explicit CUDA-event mmap lifetime guard; generation fails closed until forward lands.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - main has no GLM5.2 docs yet, so this needs a new model-line doc.
  - `feat/glm52-pp8-decode:openinfer-glm52/src/weights/load.rs` - useful H2D/header-check pattern, but it also packs expert weights and builds decode state, which this slice excludes.
  - `feat/glm52-pp8-decode:openinfer-glm52/src/lib.rs` and `src/runner.rs` - useful worker/coordinator shape, but the PP/decode/MTP runtime branches are out of scope.
- **Relevant history**:
  - No GLM5.2 history exists on latest main.
  - Remote checkpoint inspection found official `GLM-5.2-FP8` has `model.layers.0..78`, where layer 78 is MTP. Official attention shapes are `q_b_proj [16384,2048]` and `kv_b_proj [28672,512]`, not the older bad Provider 4x shapes.
- **Plan**:
  1. Add a new `openinfer-glm52` crate with config probing, safetensors manifest coverage, rank-local tensor plans, and raw H2D tensor loading.
  2. Add a load-only rank worker/coordinator that holds `CudaSlice` weights resident and returns `Rejected` for generation.
  3. Wire `openinfer-server --features glm52` detection and launch without adding kernel, DeepEP, DeepGEMM, TRTLLM, PP, or decode modules.
  4. Add an ignored real-checkpoint test using `OPENINFER_TEST_MODEL_PATH` or `models/GLM-5.2-FP8` load plus fail-closed request behavior.
  5. Verify formatting and compile/test the GLM52 feature path.
- **Risks / open questions**:
  - The ignored checkpoint test needs 8 visible GPUs and the full model path.
  - This slice proves residency and manifest coverage only; it intentionally makes no decode correctness or performance claim.

## Execution Log

### Load-Weight Crate

- Added `openinfer-glm52` with:
  - config validation for GLM5.2 FP8 constants and MTP config markers;
  - exact manifest coverage over checkpoint tensors;
  - DP1/EP8 rank plans: rank0 non-expert tensors plus experts 0..31, ranks1..7 experts 32..255 in 32-expert chunks;
  - header dtype/shape validation and coalesced `memcpy_htod` loading into rank-local slabs.

### Server Wiring

- Added workspace member and optional `openinfer-server` feature `glm52`.
- Added `model_type=glm_moe_dsa` detection.
- Added `openinfer_glm52::launch` dispatch with `--dp-size` defaulting to DP1 for GLM52, while explicit non-1 values fail validation.
- Added an explicit `bench_serving` rejection because this branch has no forward path.

### Rebase and Real-Checkpoint Load

- Fetched `origin/main` on 2026-06-30; `feat/glm52-load-weights-dp1-ep8`, `origin/main`, and local `main` all point at `1c71fee26adf9fd3a7c79855f1cc6b4238bbddc2`.
- `git rebase origin/main` was a no-op: current branch is already up to date.
- Validated the load-only command shape:

```bash
cargo run --release -p openinfer-glm52 --bin glm52_load_weights -- --model-path models/GLM-5.2-FP8 --tp-size 1 --dp-size 1
```

- Real-checkpoint load on an 8x H200 validation host completed with:
  - rank worker spawn: `3.68s`
  - rank weight H2D load: `76.82s`
  - total command elapsed: `81622ms`
  - resident bytes: rank0 `105.1 GiB`, ranks1..7 `85.5 GiB` each

### Load-Time Optimization

- The original path gave every safetensors tensor its own `CudaSlice<u8>` and issued one H2D copy per tensor. That is a poor fit for GLM5.2 because each rank loads `14.6k-16.5k` tensors.
- Changed each rank to allocate one resident GPU slab sized from the validated tensor contracts, then store tensor metadata as `name -> {offset, bytes}`.
- Within each safetensors shard, tensors are loaded in source byte order and adjacent source ranges are coalesced into one H2D copy. This keeps load-only scope: no decode kernels, no DeepEP/DeepGEMM/FlashMLA, no runtime layout packing.
- Added an explicit source-mmap lifetime guard: after each shard's async coalesced H2D copies, the loader records a CUDA event and keeps that shard mapping alive until the event completes. A one-shard event group was retained because larger groups kept too many mappings live and regressed the tail. The guard also synchronizes before dropping mappings on error paths.
- Real-checkpoint A/B on the same 8x H200 validation host:

| Version | Rank load cost | Command elapsed | Notes |
| --- | ---: | ---: | --- |
| per-tensor `CudaSlice` + per-tensor H2D | `76.82s` | `81622ms` | baseline |
| rank-local slab + coalesced H2D, no explicit mmap guard | `46.13s` | `51221ms` | diagnostic only; rejected because async copies could outlive the mapping |
| keep all shard mappings live until rank sync | `87.70s` | `91100ms` | rejected; mapping lifetime safe but slower than baseline |
| sync every 16-shard window | `74.08s` | `78770ms` | rejected; lifetime safe but barely below baseline |
| synchronous `cuMemcpyHtoD_v2` copies | `70.96s` | `75585ms` | rejected; safe but leaves too much copy overhead |
| 16-shard event groups, 4 live groups | `79.31s` | `82585ms` | rejected; large mapping groups caused tail regression |
| rank-local slab + coalesced H2D + one-shard CUDA event guard, first RAII run | `58.75s` | `63420ms` | retained implementation; explicit error-path mmap cleanup |
| rank-local slab + coalesced H2D + one-shard CUDA event guard, immediate repeat | `46.04s` | `50803ms` | retained implementation; page cache warm repeat |

- Retained-run profile:
  - first RAII run: rank0 `58.75s`, ranks1..7 `36.08-53.07s`, command `63420ms`
  - immediate repeat: rank0 `44.33s`, ranks1..7 `35.64-46.04s`, command `50803ms`
  - rank0 loads `105.1 GiB`, `16485` tensors, `3581` H2D copies
  - ranks1..7 each load `85.5 GiB`, `14592` tensors, `638-644` H2D copies

## Debrief

- **Outcome**: The branch is rebased on latest `main` and contains a load-weight-only GLM5.2 slice: config/manifest validation, DP1/EP8 rank load plans, coalesced H2D loading into rank-local slabs, server feature detection, and fail-closed generation behavior.
- **Pitfalls encountered**:
  - The first draft used the wrong DP wording in file names and tests. It is now consistently `DP1/EP8`; `--dp-size` must be `1`, while EP8 is modeled as GLM52 rank-local expert slicing.
  - The first ignored checkpoint test exposed a local machine path and duplicated logging setup. It now uses `OPENINFER_TEST_MODEL_PATH` or `models/GLM-5.2-FP8`, and uses `openinfer_core::logging::init_default()`.
  - The first slab/coalescing optimization relied on implicit async H2D/pageable staging behavior. The retained path makes the source-mmap lifetime explicit with CUDA events before releasing mappings, including early-return cleanup.
- **Verification**:
  - `cargo fmt --all --check`
  - `git diff --check`
  - `cargo check -p openinfer-glm52`
  - `cargo check -p openinfer-server --features glm52`
  - `cargo check -p openinfer-server --no-default-features --features glm52`
  - `cargo test -p openinfer-glm52 --lib`
  - `cargo test -p openinfer-server --no-default-features --features glm52 glm52_`
  - real-checkpoint load-only binary completed with `GLM5.2 load weights complete: elapsed_ms=63420`
  - immediate repeat completed with `GLM5.2 load weights complete: elapsed_ms=50803`
