# GLM5.2 TP4 GB300 Bring-Up

> **TL;DR:** Active bring-up for serving GLM-5.2-FP8 on one 4xGB300 host with `--tp-size 4`; TP4 now boots, pre-captures graphs, and serves longer, long-prompt, and bucket-8 concurrent completion smokes on GB300. Remaining work is golden/oracle coverage and perf characterization.
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` - routed this task to the GLM5.2 model-line docs and the kernel boundary.
  - `docs/models/glm52/serving-status.md` - showed the current GLM5.2 serving surface and the low-latency TP8/attention-TP arc.
  - `docs/models/glm52/dp8-scheduler.md` - explained the existing lock-step eight-rank scheduler contract.
  - `docs/models/glm52/ep8-deepep-moe.md` - captured why the original full-model path assumes 8 ranks and packed expert placement at load time.
  - `docs/models/glm52/whole-step-decode-graph.md` - captured the graph and bucket contracts that TP changes must preserve.
  - `docs/models/glm52/moe-tp8-low-latency.md` - documented the existing TP8 MoE and attention-TP topology that TP4 should generalize from.
  - `docs/subsystems/kernels/openinfer-kernels-boundary.md` - confirmed GLM5.2 kernel surfaces are model-local on top of the shared MoE/MLA substrate.
- **Relevant history**:
  - `docs/models/glm52/moe-tp8-low-latency.md` - TP8 proved that low-latency GLM uses replicated activations plus rank-sliced FP8 experts/heads, but its constants are currently 8-rank-specific.
  - `docs/models/kimi-k2/dp-design.md` - warns that TP2/TP4 shapes are not automatic once MoE and TP interact.
  - `docs/lessons/kimi-bringup-numerics.md` - greedy parity can break from small partial-sum changes, so TP4 needs runtime text/smoke evidence rather than compile-only claims.
- **Plan**:
  1. Inspect the current GLM5.2 launch, weight-loader, TP8 MoE, attention all-reduce, scheduler, and kernel build paths for fixed eight-rank assumptions.
  2. Check vendored FlashInfer first for reusable TP4 collectives, Blackwell MLA, and Blackwell grouped-GEMM paths before adding custom kernels.
  3. Add a rank-count-aware launch/topology path for the requested 4xGB300 target, starting from `--tp-size 4` and the existing TP low-latency mode rather than pretending the EP8 path can run on 4 devices.
  4. Build with `cargo build --release --features glm52` on this GB300 host, fixing compile and SM103 build blockers.
  5. Start OpenInfer against `/mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541` with `--tp-size 4` and run a minimal `/v1/completions` smoke.
- **Risks / open questions**:
  - TP4 is not a one-line validation change: existing kernels and buffers are named TP8 and bake `RANKS=8`, 8 tokens, 8 attention heads per rank, and 1/8 expert slices.
  - GB300 is SM103; the current GLM DeepGEMM AOT kernels only build real code for sm90a and become NOT_SUPPORTED stubs on SM103.
  - Full correctness needs new TP4 oracle coverage after the first runtime smoke, because partial-sum and routing orders differ from both EP8 and TP8.

## Execution Log

### Step 1: Inspect current state

- `git status --short --branch` reported a clean `main` worktree.
- `/mnt/shared` is mounted read-only, but the same commit is available writable at `/work/openinfer`; edits are being made there.
- `nvidia-smi` shows this host has 4x NVIDIA GB300, each with 284208 MiB and compute capability 10.3.
- Source inspection found:
  - `openinfer-server/src/config.rs` rejects GLM5.2 `--dp-size` values other than 8.
  - `openinfer-glm52/src/lib.rs` rejects `--tp-size != 1`.
  - `validate_startup` requires TP1/DP8/EP8 and exactly `GLM52_EP_RANKS` devices.
  - `weights.rs` defines `GLM52_EP_RANKS = 8`, and the weight manifest emits eight rank bundles.
  - The low-latency path has attention-TP and TP8 MoE code, but it is hardwired through constants such as `GLM52_TP8_RANKS = 8`, `GLM52_TP8_TOKENS = 8`, and 1/8 head/expert slicing.

### Step 2: Check FlashInfer before adding kernels

- Vendored FlashInfer has a TRT-LLM allreduce path that dispatches rank counts `{2, 4, 6, 8, 16}` in `include/flashinfer/comm/trtllm_allreduce.cuh`, and the newer allreduce-fusion path supports `{2, 4, 8, 16}`.
- FlashInfer also has Blackwell CuTe DSL GEMM+allreduce examples and tests. The test-side `can_implement` admits distributed world sizes `{2, 4, 8}`, but the code is Python/Torch-distributed symmetric-memory oriented, so it is a reference/design source rather than a direct Rust runtime dependency.
- FlashInfer has Blackwell-related MLA/GEMM surfaces (`attention/blackwell`, `cute_dsl/attention`, SM103 fused-MoE generators, and SM100/SM103 GEMM paths). These should be preferred before writing replacement GLM kernels for GB300.

### Step 3: Build current GLM5.2 on GB300

- Plain `cargo` was not on `PATH`; the usable toolchain is:
  - `CARGO_HOME=/mnt/shared/home/susun/.cargo`
  - `RUSTUP_HOME=/mnt/shared/home/susun/.rustup`
  - `/mnt/shared/home/susun/.cargo/bin/cargo`
- CUDA 13.0 nvcc lists `compute_103`.
- Initial build failed because the system NCCL is 2.28.9 and GLM MoE requires NCCL >= 2.30.4.
- The suitable NCCL root is `/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl` (2.30.7).
- Verified build command:

```bash
CARGO_HOME=/mnt/shared/home/susun/.cargo \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-server
```

- Result: build passed in `1m 27s`.
- Important warnings:
  - `No sm_90a target; GLM5.2 DeepGEMM glm52_deepgemm_grouped kernels compile as NOT_SUPPORTED stubs`
  - `No sm_90a target; GLM5.2 DeepGEMM glm52_deepgemm_mqa kernels compile as NOT_SUPPORTED stubs`

### Step 4: Run current binary to capture launch blockers

- Runtime needs the 2.30.7 NCCL library first:

```bash
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
/tmp/openinfer-target/release/openinfer --help
```

- Without that `LD_LIBRARY_PATH`, the binary picks system NCCL 2.28 and fails with `undefined symbol: ncclCommQueryProperties`.
- Current help still says `--tp-size` is for Qwen3/Kimi and "GLM5.2 requires 1"; `--moe-topo` only accepts `ep8|tp8`.
- Current TP4 command:

```bash
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
/tmp/openinfer-target/release/openinfer \
  --model-path /mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 \
  --tp-size 4 \
  --moe-topo tp8 \
  --port 18000
```

- Result: fails before loading weights with `GLM5.2 requires --tp-size=1, got 4`.

### Step 5: Add explicit TP4 topology scaffold

- Updated `openinfer-server` so GLM5.2 accepts `--moe-topo tp4`, defaults it to `--dp-size 1`, and rejects explicit non-DP1 values for that topology.
- Updated `openinfer-glm52::Glm52MoeTopo` with `Tp4`, `default_dp_size()`, and `device_count()`.
- Added a GLM launch guard for TP4 that validates `--tp-size 4 --dp-size 1` and then stops with the current real blocker instead of the stale TP1-only error.
- Focused test command:

```bash
CARGO_HOME=/mnt/shared/home/susun/.cargo \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release --features glm52 -p openinfer-server glm52_ -- --nocapture
```

- Result: 4 GLM server validation tests passed.
- TP4 launch smoke from the edited binary:

```bash
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
/tmp/openinfer-target-wip/release/openinfer \
  --model-path /mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 \
  --tp-size 4 \
  --moe-topo tp4 \
  --port 18000
```

- Result: reaches the new GLM TP4 guard and fails with:
  `GLM5.2 TP4 GB300 launch is scaffolded but not runnable yet: TP4 needs rank-count-aware MoE/attention collectives and Blackwell replacements for the current sm90a-only DeepGEMM MQA/grouped kernels`.

### Step 6: Kernel direction after FlashInfer audit

- Existing OpenInfer TP8 kernels cannot be reused by changing `nranks` alone:
  - Attention allreduce hardcodes `kRanks = 8` and `kChunk = hidden / 8`; TP4 needs `hidden / 4`.
  - MoE hardcodes 1/8 intermediate slices (`kSliceI = 256`, `kSliceRows = 512`); TP4 needs `kSliceI = 512`, `kSliceRows = 1024`.
  - Rust wrappers expose fixed `[u64; GLM52_TP8_RANKS]` peer arrays and TP8 scratch sizes.
- FlashInfer has TP4-capable collective code (`trtllm_allreduce`, `trtllm_allreduce_fusion`, and MNNVL fusion dispatches). These are the preferred source for attention/output reductions before adding a new bespoke OpenInfer collective.
- MoE/FP8 still needs a Blackwell-capable grouped GEMM path. Current SM103 builds produce `NOT_SUPPORTED` for GLM DeepGEMM grouped and MQA kernels, so a full server smoke cannot pass until those are replaced or bypassed.

### Step 7: Thread TP4 through host-side topology plumbing

- Added topology helpers on `Glm52MoeTopo`:
  - `Tp4` expects TP4/DP1/EP1 and four devices.
  - `Tp8` and `Tp4` are both classified as tensor-replicated MoE topologies.
- Updated startup validation and rank-bundle generation so topology rank count is not always `GLM52_EP_RANKS`:
  - EP8 still creates eight expert-owning rank bundles.
  - TP8 still creates eight non-expert replicated bundles.
  - TP4 can now request four non-expert replicated bundles.
- Generalized the host TP slice staging loader:
  - TP8 remains 1/8 slices (`slice_i=256`, `slice_rows=512`).
  - TP4 stages 1/4 slices (`slice_i=512`, `slice_rows=1024`).
  - The TP8 CUDA launcher now rejects non-TP8 slice geometry explicitly instead of silently accepting a TP4 bank.
- Updated model construction so tensor-replicated attention sharding is based on topology rank count:
  - TP8 keeps 8 MLA heads per rank.
  - TP4 keeps 16 MLA heads per rank.
- Verified:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release -p openinfer-glm52 slice_staging -- --nocapture
```

- Result: 2 slice staging tests passed, including the new TP4 geometry test.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release --features glm52 -p openinfer-server glm52_ -- --nocapture
```

- Result: 4 GLM server validation tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-server
```

- Result: build passed in `1m 22s`. The SM103 build still emits `NOT_SUPPORTED` warnings for the sm90a-only GLM DeepGEMM grouped/MQA kernels.
- TP4 smoke still intentionally stops at the GLM TP4 guard; it now does so after tokenizer/frontend setup and before weight load.

### Step 8: Move TP4 guard past model build, before unsafe collective setup

- Removed the early `launch()` bail for `Glm52MoeTopo::Tp4`.
- Updated `build_rank_models` so TP4 stops after all rank models build and before any collective context creation. This is deliberate: the old next step would have called `Glm52MoeEp8State::new(..., GLM52_EP_RANKS=8, rank)` from only four workers, which risks an NCCL/DeepEP collective init hang.
- The new TP4 failure boundary is:
  `GLM5.2 TP4 loaded weights and built the four sharded rank models, but serving is not runnable yet: TP4 still needs CUDA attention/MoE collective state and a Blackwell-capable indexer MQA path before entering decode`.
- Updated scheduler mirroring so future TP4 decode uses the same one-logical-rank mirrored scheduling branch as TP8 once CUDA state exists.
- Added topology tests for the intended shapes:
  - TP4 = TP4/DP1/EP1, four devices, tensor-replicated MoE.
  - EP8/TP8 keep their existing TP1/DP8/EP8 eight-device shapes.
- Verified:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release -p openinfer-glm52 topology -- --nocapture
```

- Result: 2 topology tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release -p openinfer-glm52 slice_staging -- --nocapture
```

- Result: 2 slice staging tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release --features glm52 -p openinfer-server glm52_ -- --nocapture
```

- Result: 4 GLM server validation tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-server
```

- Result: build passed in `1m 20s`; SM103 still emits the same `NOT_SUPPORTED` warnings for GLM DeepGEMM grouped/MQA.
- Full TP4 launch was not run to the new post-build guard in this step because it would load and build the full four-rank checkpoint only to hit the known missing CUDA TP4 collective/indexer boundary.

### Step 9: Add TP4 CUDA collectives and reach the FlashMLA GB300 boundary

- Added TP4 kernel entry points by specializing the existing TP8 low-latency kernels:
  - `openinfer-kernels/csrc/glm52/glm52_moe_tp4.cu`
  - `openinfer-kernels/csrc/glm52/glm52_tp4_ar.cu`
  - `openinfer-kernels/csrc/glm52/glm52_tp4_ll.cuh`
  - `openinfer-kernels/src/ops/glm52/moe_tp4.rs`
  - `openinfer-kernels/src/ops/glm52/tp4_ar.rs`
- TP4 geometry differs from TP8:
  - ranks = 4
  - attention AR chunk = `6144 / 4 = 1536` bf16
  - expert intermediate slice = `2048 / 4 = 512`
  - gate/up slice rows = `1024`
  - replicated token rows remain 8, so `UNION_MAX = 8 * (topk + shared)`, not `ranks * (...)`.
- Registered TP4 FFI symbols and exports alongside TP8.
- Fixed a link conflict from the copied CUDA VMM allocator state by giving the TP4 globals distinct names.
- Generalized the TP LL rendezvous so one exchange can wait for either four or eight ranks.
- Added a TP runtime enum:
  - TP8 uses the existing eight-rank state.
  - TP4 allocates four-rank LL windows and dispatches to the new TP4 MoE and attention AR wrappers.
- Changed setup so:
  - EP8 creates the DeepEP state.
  - TP8/TP4 skip DeepEP and create tensor-replicated TP state.
  - decode treats EP8 state as optional and only requires it if an EP8 MoE layer is actually reached.
- Removed the post-model-build TP4 guard.
- Verified:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release -p openinfer-glm52 topology -- --nocapture
```

- Result: 2 topology tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release -p openinfer-glm52 slice_staging -- --nocapture
```

- Result: 2 slice-staging tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release --features glm52 -p openinfer-server glm52_ -- --nocapture
```

- Result: 4 GLM server validation tests passed.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-server
```

- Result: build passed in `1m 15s`; SM103 still emits the known `NOT_SUPPORTED` warnings for GLM DeepGEMM grouped/MQA.
- TP4 launch with `--max-model-len 1024` failed before model build because GLM5.2 enforces a minimum cap of 4096.
- TP4 launch with the minimum cap:

```bash
timeout 420s env \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/tmp/openinfer-target-wip/release/openinfer \
  --model-path /mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 \
  --tp-size 4 \
  --moe-topo tp4 \
  --max-model-len 4096 \
  --port 18080
```

- Result: loaded all four non-expert replicated rank weight sets, then failed during model build with:
  `GLM5.2 FlashMLA sparse num_sm_parts query failed: DriverError(CUDA_ERROR_NOT_SUPPORTED, "operation not supported")`.
- FlashInfer/FlashMLA check:
  - The current OpenInfer wrapper `glm52_flashmla_sparse.cu` includes the SM90 sparse FP8 FlashMLA path and explicitly returns `CUDA_ERROR_NOT_SUPPORTED` unless the current device is SM90.
  - Vendored FlashMLA already has `csrc/sm100/decode/head64` with a V32 instantiation, which matches GLM5.2's 64-head, 576-QK, 512-V sparse-decode shape better than writing a new OpenInfer attention kernel from scratch.
  - Vendored FlashInfer proper also has Blackwell MLA/TRTLLM sparse MLA infrastructure, but it is exposed through the Python/JIT/TVM FFI stack rather than this Rust FFI wrapper.
  - Next kernel work should wire the existing Blackwell FlashMLA/FlashInfer sparse MLA implementation into `glm52_flashmla_sparse.cu` or replace the wrapper with a FlashInfer-backed FFI, before inventing a new kernel.

### Step 10: Wire FlashMLA SM100 sparse decode and expose the DeepGEMM MQA blocker

- Updated `glm52_flashmla_sparse.cu` to mirror FlashMLA's upstream sparse-decode dispatch:
  - SM90 keeps the existing `sm90::decode::sparse_fp8` path.
  - SM100-family devices use `sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32>`.
  - `num_sm_parts` now accepts SM100-family devices and uses the upstream head64 formula `max(num_sms / s_q, 1)`.
- Updated `openinfer-kernels/build.rs` so only the `glm52_flashmla_sparse.cu` TU promotes SM100-family targets to FlashMLA's `compute_100f,code=sm_100f`. Compiling that TU as plain `sm_103` failed because ptxas rejects `tcgen05`/CTA-group features on target `sm_103`.
- Kept `glm52_trtllm_grouped_fp8.cu` on the normal GLM FlashMLA arch path; only sparse decode needs `sm_100f` today.
- Verified kernel and server builds:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-kernels
```

- Result: build passed in `1m 12s`; build log shows `Compiling GLM5.2 FlashMLA sparse decode for nvcc targets: sm_100f`.

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-server
```

- Result: build passed in `1m 25s`.
- TP4 launch with `--max-model-len 4096` now loads weights, builds rank models, creates TP4 contexts, and starts the OpenAI server on port 18080.
- Minimal completion request:

```bash
curl -sS --max-time 180 http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541","prompt":"Hello","max_tokens":4,"temperature":0}'
```

- Result: the request entered decode graph pre-capture and failed at the next GB300 blocker:
  `GLM5.2 graph pre-capture failed: GLM5.2 graph pre-capture (bucket 8, full_tier false) on rank 0: GLM5.2 layer 0 attention half: GLM5.2 DeepGEMM MQA metadata launch failed: DriverError(CUDA_ERROR_NOT_SUPPORTED, "operation not supported")`.
- The HTTP client did not receive a response after the server-side pre-capture error and was interrupted; the server then logged an auto-abort for the dropped request stream. That is a follow-up error-propagation issue, separate from the kernel blocker.

### Step 11: Add SM100 DeepGEMM MQA and fix TP4 scratch sizing

- Added a Blackwell path in `openinfer-kernels/csrc/glm52/glm52_deepgemm_mqa.cu` using DeepGEMM's SM100 paged-MQA logits metadata and logits kernels. `openinfer-kernels/build.rs` now compiles that translation unit as `sm_100f` on SM100-family targets when nvcc accepts the target.
- Added TP4 attention GEMV allow-list shapes:
  - `n=4096, k=2048` for the TP4 `q_b` attention shard.
  - `n=6144, k=4096` for the TP4 `o_proj` attention shard.
- Fixed a TP4 MoE scratch-buffer overrun. The CUDA kernel indexes `guprob`, `bpart`, `ug`, and `cpart` by the eight replicated rows (`kTokens=8`), but the Rust-side TP4 allocation had kept rank-count sizing (`RANKS=4`) for several buffers. The final host sizes now use `GLM52_TP4_TOKENS` for `BPART`, `GUPROB`, `UG`, and `CPART`.
- Diagnostic path:
  - Before the scratch fix, `"Hello"` could return because it stopped without generating tokens, while `"The capital of France is"` returned HTTP 500 with `greedy argmax found no finite logit`.
  - A finite probe with graph replay disabled showed first-step logits were finite; the second generated step reached layer 1 with finite post-attention residuals but non-finite `normed2`, consistent with earlier memory corruption rather than an RMSNorm math bug.
  - After the scratch fix, the temporary finite probes and graph-disable hook were removed and the fused add+RMSNorm path was restored.
- Final verified build:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_BUILD_TIMING=1 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
OPENINFER_NVCC_JOBS=16 \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo build --release --features glm52 -p openinfer-server
```

- Result: build passed; the build log shows `Compiling GLM5.2 FlashMLA sparse decode for nvcc targets: sm_100f`.
- Final verified server launch:

```bash
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/tmp/openinfer-target-wip/release/openinfer \
  --model-path /mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541 \
  --tp-size 4 \
  --moe-topo tp4 \
  --max-model-len 4096 \
  --port 18080
```

- Result: server loaded all four TP4 rank models, started the OpenAI endpoint, and logged `GLM5.2 whole-step graphs pre-captured: 1 buckets x 2 tiers`.
- Final HTTP smokes:

```bash
curl -sS --max-time 180 http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541","prompt":"Hello","max_tokens":4,"temperature":0}'
```

- Result: returned `"Hello"` with `finish_reason="stop"` and `completion_tokens=0`.

```bash
curl -sS --max-time 180 http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541","prompt":"The capital of France is","max_tokens":4,"temperature":0}'
```

- Result: returned `" Paris. The population"` with `finish_reason="length"` and `completion_tokens=4`. Server logs showed both requests completed cleanly.
- Cleanup checks:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release --features glm52 -p openinfer-server glm52_ -- --nocapture
```

- Result: 4 GLM server validation tests passed.

```bash
git diff --check
```

- Result: no whitespace errors.

### Step 12: Strengthen TP4 serving smoke coverage

- Reused the verified TP4 server command from Step 11 on the GB300 host. It loaded four rank models, started the OpenAI endpoint, and pre-captured `1 buckets x 2 tiers`.
- Longer single-request decode:

```bash
curl -sS --max-time 240 http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/mnt/shared/home/susun/hf_cache/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541","prompt":"Write a concise paragraph about why tensor parallel inference needs synchronized collectives.","max_tokens":32,"temperature":0}'
```

- Result: HTTP 200, `prompt_tokens=14`, `completion_tokens=32`, `finish_reason="length"`.
- Four-way concurrent decode via a Python `urllib` thread pool:
  - Result: all 4 requests returned HTTP 200, `completion_tokens=16`, `finish_reason="length"`.
- Bucket-8 concurrent decode via the same Python harness:
  - Result: all 8 requests returned HTTP 200, `completion_tokens=16`, `finish_reason="length"`.
- Long-prompt request:

```text
prompt_tokens=874, completion_tokens=16, finish_reason="length"
```

- Server logs for the longer, four-way, eight-way, and long-prompt requests showed only normal `completion finished` records; no rank-side errors, aborts, or finite-logit failures appeared.
- Focused GLM5.2 crate tests:

```bash
CARGO_HOME=/tmp/openinfer-cargo-home \
RUSTUP_HOME=/mnt/shared/home/susun/.rustup \
CARGO_TARGET_DIR=/tmp/openinfer-target-wip \
OPENINFER_CUDA_SM=103 \
OPENINFER_NCCL_ROOT=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl \
LD_LIBRARY_PATH=/mnt/shared/home/susun/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH \
PATH=/mnt/shared/home/susun/.cargo/bin:/usr/local/cuda/bin:$PATH \
/mnt/shared/home/susun/.cargo/bin/cargo test --release -p openinfer-glm52 tp4 -- --nocapture
```

- Result: 2 tests passed:
  - `topology_tests::tp4_topology_shape_is_four_rank_replicated_tp`
  - `moe_tp8::tests::tp4_slice_staging_geometry`

## Debrief

- **Outcome**: TP4 GB300 now reaches real serving coverage beyond a minimal smoke. The CLI/launch layer, host-side topology plumbing, TP4 CUDA MoE/attention collectives, four-rank TP runtime setup, FlashMLA SM100 sparse-decode dispatch, and SM100 DeepGEMM MQA path build on GB300. The cleaned graph-mode server pre-captures and completes longer, long-prompt, four-way, and bucket-8 concurrent `/v1/completions` requests.
- **Pitfalls encountered**:
  - The requested model path is the Hugging Face cache wrapper; the usable checkpoint directory is the snapshot under `snapshots/ba978f7d347eaf65d22f1a86833408afdb953541`.
  - `/mnt/shared` is read-only in this session; use `/work/openinfer` for edits.
  - Runtime must prefer NCCL 2.30.7 from the local venv over system NCCL 2.28.9.
  - FlashMLA sparse decode needs `sm_100f` SASS on GB300; compiling the SM100 implementation as plain `sm_103` fails in ptxas on `tcgen05`/CTA-group features.
  - Four-rank TP4 must never call the current DeepEP setup with `GLM52_EP_RANKS=8`; TP4 now skips DeepEP and uses the TP LL rendezvous instead.
  - TP4 row count is not the same as TP4 rank count. Several TP4 MoE scratch buffers must be sized by `kTokens=8`, not `RANKS=4`; the wrong host sizes caused memory corruption that surfaced only after the first generated token.
- **Lessons learned**:
  - TP4 bring-up had three independent workstreams: topology generalization from TP8 hardcoding, Blackwell sparse MLA/FlashMLA support, and Blackwell replacement for the sm90a-only MQA path. All three are now far enough to pass a serving smoke.
  - When generalizing TP8 code to TP4, audit kernel indexing expressions, not just the named constants. TP8 masked `RANKS == TOKENS == 8`; TP4 exposes every accidental substitution.
- **Follow-ups**:
  - Add a focused TP4 oracle or golden gate beyond HTTP smokes, because serving liveness does not prove numerical parity.
  - Run a longer soak and perf characterization after the correctness gate exists.
  - Decide whether to replace the remaining grouped-kernel warning with a true SM100 path or keep it as an unused EP8/DeepGEMM build warning for TP4.
