# DFlash Model Download

> **TL;DR:** `z-lab/Qwen3-4B-DFlash-b16` is downloaded and verified locally at `/data/models/Qwen3-4B-DFlash-b16` and on the 5090 box at `/data/Qwen3-4B-DFlash-b16`; it is the drafter artifact for Qwen3-4B speculative decoding bring-up.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - Qwen3 model-line docs live under `docs/models/qwen3/`.
  - `docs/models/qwen3/model-crate.md` - Qwen3 runtime is model-crate owned; local model artifacts under `/data/models` are used by executor/tests through model paths.
- **Relevant history**:
  - No existing DFlash or speculative-decoding task doc found.
- **Plan**:
  1. Check `/data/models` capacity and whether the target directory already exists.
  2. Download `z-lab/Qwen3-4B-DFlash-b16` with the Hugging Face Hub CLI into `/data/models/Qwen3-4B-DFlash-b16`.
  3. List the downloaded files and verify the safetensors/config metadata is present.
- **Risks / open questions**:
  - The Hugging Face repo may require auth or may use custom code files that are not part of the plain safetensors load path yet.

## Execution Log

### Step 1: Check destination and capacity
- `/data/models` already contains the base `/data/models/Qwen3-4B` artifact.
- `df -h /data/models /data` reported `753G` available on `/data`, enough for the DFlash drafter.
- Target path chosen: `/data/models/Qwen3-4B-DFlash-b16`.

### Step 2: Download from Hugging Face
- Command:
  ```bash
  uvx --from huggingface_hub hf download z-lab/Qwen3-4B-DFlash-b16 --local-dir /data/models/Qwen3-4B-DFlash-b16
  ```
- Result: fetched `9` files, `1.08GB` total, into `/data/models/Qwen3-4B-DFlash-b16`.

### Step 3: Verify local files
- `find /data/models/Qwen3-4B-DFlash-b16 -maxdepth 2 -type f` shows:
  - `config.json`
  - `model.safetensors`
  - `modeling_dflash.py`
  - `dflash.py`
  - `utils.py`
  - `README.md`
  - `.gitattributes`
  - `assets/dflash_system.png`
  - `assets/speedup.png`
- `du -sh /data/models/Qwen3-4B-DFlash-b16` reports `1.1G`.
- `jq` parsed `config.json`; `architectures = ["DFlashDraftModel"]`, `hidden_size = 2560`, `num_hidden_layers = 5`, `vocab_size = 151936`.
- `safetensors` opened `model.safetensors` successfully and reported `58` tensors.

### Step 4: Place artifact on 5090
- User explicitly approved placing a copy under `/data` on the 5090 box.
- Remote path: `/data/Qwen3-4B-DFlash-b16`.
- Download command used the 5090 proxy from root's `.bashrc`:
  ```bash
  export http_proxy=http://172.17.0.1:1081
  export https_proxy=http://172.17.0.1:1081
  hf download z-lab/Qwen3-4B-DFlash-b16 --local-dir /data/Qwen3-4B-DFlash-b16 --max-workers 8
  ```
- The 5090 copy contains the same core files as the local copy; `model.safetensors` is `1074860568` bytes.
- Real-weight validation now passes on 5090 with:
  ```bash
  OPENINFER_TEST_MODEL_PATH=/data/Qwen3-4B OPENINFER_DFLASH_TEST_MODEL_PATH=/data/Qwen3-4B-DFlash-b16 cargo test --release -p openinfer-qwen3-4b dflash::tests::downloaded_dflash_config_matches_qwen3_4b --lib -- --nocapture
  ```

## Debrief

- **Outcome**: The DFlash drafter model is present at `/data/models/Qwen3-4B-DFlash-b16` locally and `/data/Qwen3-4B-DFlash-b16` on 5090.
- **Pitfalls encountered**:
  - None during download. The repo includes Python custom-code files, so runtime integration still needs a native Rust loader/forward path rather than relying on `trust_remote_code`.
- **Lessons learned**:
  - The artifact is small enough (`1.1G`) to keep alongside the base Qwen3-4B model.
  - `config.json` does not set `torch_dtype`; integration should infer/check tensor dtype from safetensors rather than trusting that config field.
